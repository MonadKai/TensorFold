//! A transcription's bounded child requests use the shared engine and retain their original chunk order.
const std = @import("std");
const json = @import("json");
const audio = @import("audio_input.zig");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;

pub const Span = struct { start: usize, end: usize };
pub const Output = struct { text: []const u8, chunks: []json.Value };
const Result = struct { runtime: ?[]u8 = null, text: ?[]u8 = null, failure: ?chat.Failure = null, kind: errors.Kind = .request, message: ?[]u8 = null };
const State = struct {
    srv: *Server,
    samples: []const f32,
    spans: []const Span,
    results: []Result,
    context: []const u8,
    lang: ?[]const u8,
    fields: json.Value,
    max_tokens: i64,
    temperature: f64,
    next: std.atomic.Value(usize) = .init(0),
    left: std.atomic.Value(usize),
    cancelled: std.atomic.Value(bool) = .init(false),

    pub fn check(state: *State) bool {
        return state.cancelled.load(.acquire);
    }

    fn work(state: *State) void {
        defer _ = state.left.fetchSub(1, .release);
        while (!state.check()) {
            const i = state.next.fetchAdd(1, .monotonic);
            if (i >= state.spans.len) break;
            state.one(i) catch |e| {
                state.results[i].failure = e;
                state.cancelled.store(true, .release);
                break;
            };
        }
    }

    fn one(state: *State, index: usize) chat.Failure!void {
        const profile = std.c.getenv("TF_ASR_PROFILE") != null;
        const clock = @import("clock.zig");
        const queued = if (profile) clock.nowNs(state.srv.io) else 0;
        const gate = &state.srv.audio_children;
        try gate.wait(state.srv.io, @import("audio_quota.zig").Gate.limit(state.srv.info.lanes), state);
        defer gate.release(state.srv.io);
        const admission_seconds = if (profile) clock.seconds(clock.nowNs(state.srv.io) - queued) else 0;
        var arena = std.heap.ArenaAllocator.init(state.srv.gpa);
        defer arena.deinit();
        var cx: errors.Cx = .{ .a = arena.allocator() };
        const span = state.spans[index];
        const result = &state.results[index];
        errdefer {
            result.kind = cx.kind;
            result.message = state.srv.gpa.dupe(u8, cx.message) catch null;
        }
        const rendered = try audio.prepare(state.srv, &cx, state.samples[span.start..span.end], state.context, state.lang, false);
        const input: chat.Input = .{ .prompt = .{ .ids = rendered.ids }, .audios = rendered.audios, .fields = state.fields, .max_tokens = state.max_tokens, .temperature = state.temperature };
        const reply = try chat.run(state.srv, &cx, input, null, state);
        result.text = try state.srv.gpa.dupe(u8, reply.content);
        if (profile) {
            try reply.runtime.object.put(cx.a, "audio_admission_seconds", .{ .float = admission_seconds });
            result.runtime = try state.srv.gpa.dupe(u8, try json.stringify(cx.a, reply.runtime, .{}));
        }
    }
};

pub fn run(srv: *Server, cx: *errors.Cx, samples: []const f32, spans: []const Span, context: []const u8, lang: ?[]const u8, fields: json.Value, max_tokens: i64, temperature: f64, no_space: bool, gone: anytype) chat.Failure!Output {
    const results = try srv.gpa.alloc(Result, spans.len);
    defer srv.gpa.free(results);
    @memset(results, .{});
    defer for (results) |result| {
        if (result.text) |text| srv.gpa.free(text);
        if (result.runtime) |runtime| srv.gpa.free(runtime);
        if (result.message) |message| srv.gpa.free(message);
    };
    const count = @min(@max(srv.info.lanes / 2, 1), 2);
    var state: State = .{ .srv = srv, .samples = samples, .spans = spans, .results = results, .context = context, .lang = lang, .fields = fields, .max_tokens = max_tokens, .temperature = temperature, .left = .init(count) };
    var threads: [2]std.Thread = undefined;
    var started: usize = 0;
    defer {
        state.cancelled.store(true, .release);
        for (threads[0..started]) |thread| thread.join();
    }
    for (threads[0..count]) |*thread| {
        thread.* = std.Thread.spawn(.{}, State.work, .{&state}) catch return error.OutOfMemory;
        started += 1;
    }
    var disconnected = false;
    while (state.left.load(.acquire) != 0) {
        if (gone.check()) {
            disconnected = true;
            state.cancelled.store(true, .release);
        }
        std.Io.sleep(srv.io, .fromMilliseconds(5), .awake) catch {};
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    if (disconnected) return error.Cancelled;
    for (results) |result| if (result.failure) |e| if (e != error.Cancelled) {
        cx.kind = result.kind;
        cx.message = try cx.a.dupe(u8, result.message orelse "transcription child failed");
        return e;
    };
    var text: std.ArrayList(u8) = .empty;
    var telemetry: std.ArrayList(json.Value) = .empty;
    for (results, 0..) |result, i| {
        const raw = result.text orelse return error.Cancelled;
        if (result.runtime) |runtime| try telemetry.append(cx.a, (try json.parse(cx.a, runtime)).ok);
        const clean = audio.clean(raw);
        if (i != 0 and !no_space) try text.append(cx.a, ' ');
        try text.appendSlice(cx.a, clean);
    }
    return .{ .text = text.items, .chunks = telemetry.items };
}
