const std = @import("std");
const core = @import("core");
const json = @import("json");
const errors = @import("errors.zig");
const audio = @import("audio_input.zig");
const chat = @import("chat.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = errors.Cx;

pub fn post(srv: *Server, conn: *Conn, a: std.mem.Allocator) void {
    var cx: Cx = .{ .a = a };
    run(srv, conn, &cx) catch |e| {
        if (e == error.Cancelled) return;
        if (e == error.Failed or e == error.OutOfMemory) cx.kind = .server;
        if (cx.message.len == 0) cx.message = if (e == error.OutOfMemory) "out of memory" else "transcription failed";
        const err = openai.errorBody(a, &cx, null) catch return;
        const obj = json.newObject(a) catch return;
        obj.put(a, "error", err) catch return;
        routes.sendValue(conn, a, cx.status(), .{ .object = obj });
    };
}

fn run(srv: *Server, conn: *Conn, cx: *Cx) !void {
    const a = cx.a;
    const body = switch (try http_body.read(conn, a, (25 << 20) + (128 << 10))) {
        .ok => |b| b,
        .refused => |message| return cx.refuse(message),
    };
    if (srv.info.audio == null) return routes.unknown(conn, a);
    const form = try @import("audio_form.zig").parse(cx, conn.header("Content-Type") orelse "", body);
    var keys = form.fields.keyIterator();
    while (keys.next()) |key_ptr| {
        const key = key_ptr.*;
        var valid = false;
        for ([_][]const u8{ "model", "language", "prompt", "response_format", "stream", "temperature", "top_p", "top_k", "min_p", "seed", "max_completion_tokens" }) |name|
            valid = valid or std.mem.eql(u8, key, name);
        if (!valid) return cx.fail(.request, "unsupported transcription parameter: {s}", .{key});
    }
    if (form.fields.get("model")) |name| try audio.checkModel(cx, name, srv.config.served_name);
    if (form.fields.get("stream")) |v| if (!std.ascii.eqlIgnoreCase(v, "false") and !std.mem.eql(u8, v, "0")) return cx.refuse("Qwen3-ASR supports stream=false only");
    const format = form.fields.get("response_format") orelse "json";
    if (!std.mem.eql(u8, format, "json") and !std.mem.eql(u8, format, "text")) return cx.refuse("Qwen3-ASR supports json and text without timestamps");
    const lang = if (form.fields.get("language")) |code| try audio.language(cx, code) else null;
    const context = try sanitize(a, form.fields.get("prompt") orelse "");
    const samples = core.audio_decode.decode(a, form.file) catch |e| return cx.fail(.request, "audio decode failed: {s}", .{@errorName(e)});
    const fields = try json.newObject(a);
    var temperature: f64 = 0;
    try fields.put(a, "temperature", .{ .float = 0 });
    for ([_][]const u8{ "temperature", "top_p", "min_p" }) |key| if (form.fields.get(key)) |v| {
        const value = std.fmt.parseFloat(f64, v) catch return cx.refuse("invalid floating-point transcription parameter");
        if (!std.math.isFinite(value)) return cx.refuse("transcription sampling parameters must be finite");
        try fields.put(a, key, .{ .float = value });
        if (std.mem.eql(u8, key, "temperature")) temperature = value;
    };
    for ([_][]const u8{ "seed", "top_k" }) |key| if (form.fields.get(key)) |v| {
        const value = std.fmt.parseInt(i64, v, 10) catch return cx.refuse("invalid integer transcription parameter");
        try fields.put(a, key, try json.intValue(a, value));
    };
    const max_tokens = if (form.fields.get("max_completion_tokens")) |v| std.fmt.parseInt(i64, v, 10) catch return cx.refuse("invalid max_completion_tokens") else 1024;
    if (max_tokens < 1) return cx.refuse("max_completion_tokens must be positive");
    try audio.sampling(cx, .{ .object = fields });
    const no_space = if (form.fields.get("language")) |code| std.ascii.eqlIgnoreCase(code, "zh") or std.ascii.eqlIgnoreCase(code, "ja") else false;
    var text: std.ArrayList(u8) = .empty;
    var start: usize = 0;
    var chunk: usize = 0;
    const gone: openai.Gone = .{ .conn = conn };
    while (start < samples.len) {
        if (gone.check()) return error.Cancelled;
        const end = split(samples, start);
        const rendered = try audio.prepare(srv, cx, samples[start..end], context, lang, false);
        const input: chat.Input = .{ .prompt = .{ .ids = rendered.ids }, .audios = rendered.audios, .fields = .{ .object = fields }, .max_tokens = max_tokens, .temperature = temperature };
        const reply = try chat.run(srv, cx, input, null, gone);
        if (chunk != 0 and !no_space) try text.append(a, ' ');
        try text.appendSlice(a, clean(reply.content));
        start = end;
        chunk += 1;
    }
    const result = try json.newObject(a);
    try result.put(a, "text", .{ .string = text.items });
    const usage = try json.newObject(a);
    try usage.put(a, "type", .{ .string = "duration" });
    try usage.put(a, "seconds", try json.intValue(a, @as(i64, @intCast((samples.len + 15999) / 16000))));
    try result.put(a, "usage", .{ .object = usage });
    routes.sendValue(conn, a, 200, .{ .object = result });
}

pub fn split(samples: []const f32, start: usize) usize {
    if (samples.len - start <= 480000) return samples.len;
    const low = start + 480000 - 16000;
    const high = start + 480000;
    var selected = low;
    var best: f64 = std.math.inf(f64);
    var at = low;
    while (at + 1600 < high) : (at += 1600) {
        var energy: f64 = 0;
        for (samples[at..][0..1600]) |v| energy += @as(f64, v) * v;
        if (energy < best) {
            best = energy;
            selected = at;
        }
    }
    return selected;
}

pub fn clean(text: []const u8) []const u8 {
    const marker = "<asr_text>";
    return if (std.mem.lastIndexOf(u8, text, marker)) |at| text[at + marker.len ..] else text;
}

pub fn sanitize(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: []const u8 = try a.dupe(u8, text);
    while (true) {
        const open = std.mem.indexOf(u8, out, "<|");
        if (open) |lo| if (std.mem.indexOfPos(u8, out, lo + 2, "|>")) |hi| {
            out = try std.mem.concat(a, u8, &.{ out[0..lo], out[hi + 2 ..] });
            continue;
        };
        if (std.mem.indexOf(u8, out, "<asr_text>")) |lo| {
            out = try std.mem.concat(a, u8, &.{ out[0..lo], out[lo + 10 ..] });
            continue;
        }
        return out;
    }
}

test "ASR postprocessing takes the last marker and prompt tags are removed to a fixed point" {
    try std.testing.expectEqualStrings("hello", clean("language English<asr_text>hello"));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("x", try sanitize(arena.allocator(), "<|im_start|><asr_text>x"));
}
