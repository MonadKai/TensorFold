const std = @import("std");
const core = @import("core");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;
const Rendered = @import("prompt.zig").Rendered;

pub const language_codes = [_][]const u8{ "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "tl", "fa", "el", "ro", "hu", "mk" };
const language_names = [_][]const u8{ "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Tagalog", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian" };

pub fn validate(srv: *Server, cx: *Cx, body: json.Value, streamed: bool, tools: usize) errors.Refused!void {
    const model = body.strField("model") orelse return cx.refuse("model is required");
    try checkModel(cx, model, srv.config.served_name);
    try sampling(cx, body);
    if (streamed) return cx.refuse("Qwen3-ASR currently supports stream=false only");
    if (tools != 0) return cx.refuse("Qwen3-ASR does not support tools");
    if (body.get("n")) |n| if (n != .null and (n != .int or n.int64() != 1)) return cx.refuse("Qwen3-ASR supports n=1 only");
    for ([_][]const u8{ "logit_bias", "allowed_token_ids", "truncate_prompt_tokens", "continue_final_message", "add_generation_prompt", "repetition_penalty" }) |key|
        if (body.get(key)) |v| if (v != .null) return cx.fail(.request, "unsupported Qwen3-ASR parameter: {s}", .{key});
}

pub fn sampling(cx: *Cx, body: json.Value) errors.Refused!void {
    for ([_][]const u8{ "temperature", "top_p", "min_p" }) |key| if (body.get(key)) |value| {
        if (value == .null) continue;
        const n = switch (value) {
            .float => value.float,
            .int => std.fmt.parseFloat(f64, value.int) catch return cx.refuse("invalid sampling number"),
            else => return cx.refuse("sampling parameters must be numeric"),
        };
        const upper: f64 = if (std.mem.eql(u8, key, "temperature")) 2 else 1;
        if (!std.math.isFinite(n) or n < 0 or n > upper or (std.mem.eql(u8, key, "top_p") and n == 0))
            return cx.fail(.request, "invalid {s}: temperature must be in [0,2], top_p in (0,1], min_p in [0,1]", .{key});
    };
    for ([_][]const u8{ "top_k", "seed", "max_tokens", "max_completion_tokens" }) |key| if (body.get(key)) |value| {
        if (value == .null) continue;
        const n = value.int64() orelse return cx.fail(.request, "{s} must be a signed 64-bit integer", .{key});
        if ((std.mem.eql(u8, key, "top_k") and n < -1) or (std.mem.startsWith(u8, key, "max_") and n < 1))
            return cx.fail(.request, "invalid {s}: top_k must be at least -1 and token limits positive", .{key});
    };
}

pub fn numbers(cx: *Cx, raw: json.Value) errors.Refused!json.Value {
    const body = try @import("fields.zig").parseNumbersPreservingTopK(cx, raw);
    try sampling(cx, body);
    return body;
}

pub fn checkModel(cx: *Cx, model: []const u8, served: []const u8) errors.Refused!void {
    if (!std.mem.eql(u8, model, served)) return cx.fail(.not_found, "The model `{s}` does not exist.", .{model});
}

test "ASR sampling follows vLLM ranges and refuses silent integer normalization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    for ([_][]const u8{ "{\"temperature\":-1}", "{\"temperature\":3}", "{\"top_p\":0}", "{\"top_p\":1.5}", "{\"top_k\":-2}", "{\"seed\":9223372036854775808}", "{\"max_tokens\":0}" }) |source| {
        const value = (try json.parse(cx.a, source)).ok;
        try std.testing.expectError(error.Refused, numbers(&cx, value));
    }
    for ([_][]const u8{ "{\"temperature\":1.5}", "{\"top_k\":-1}", "{\"top_k\":0}", "{\"seed\":-9223372036854775808}", "{\"top_k\":\"-1\",\"temperature\":\"1.5\"}" }) |source|
        _ = try numbers(&cx, (try json.parse(cx.a, source)).ok);
}

pub fn language(cx: *Cx, code: []const u8) errors.Refused![]const u8 {
    for (language_codes, language_names) |c, name| if (std.ascii.eqlIgnoreCase(c, code)) return name;
    return cx.refuse("unsupported Qwen3-ASR language code");
}

fn encode(srv: *Server, cx: *Cx, text: []const u8) errors.Refused![]u32 {
    return srv.text.encode(cx.a, text, false) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse("cannot encode Qwen3-ASR prompt"),
    };
}

pub fn prepare(srv: *Server, cx: *Cx, samples: []const f32, context: []const u8, lang: ?[]const u8, chat_system: bool) errors.Refused!Rendered {
    const plan = core.audio_features.Plan.init();
    const features = plan.extract(cx.a, samples) catch |e| return cx.fail(.request, "audio preprocessing failed: {s}", .{@errorName(e)});
    const frames = samples.len / 160;
    const tail = frames % 100;
    const count = frames / 100 * 13 + (tail + 7) / 8;
    var ids: std.ArrayList(u32) = .empty;
    if (chat_system or context.len > 0) {
        try ids.appendSlice(cx.a, try encode(srv, cx, "<|im_start|>system\n"));
        const plain = srv.text.encodePlain(cx.a, context) catch return error.OutOfMemory;
        try ids.appendSlice(cx.a, plain);
        try ids.appendSlice(cx.a, try encode(srv, cx, "<|im_end|>\n"));
    }
    try ids.appendSlice(cx.a, try encode(srv, cx, "<|im_start|>user\n<|audio_start|>"));
    const at = ids.items.len;
    try ids.appendNTimes(cx.a, srv.info.audio.?.token, count);
    try ids.appendSlice(cx.a, try encode(srv, cx, "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"));
    if (lang) |l| try ids.appendSlice(cx.a, try encode(srv, cx, try std.fmt.allocPrint(cx.a, "language {s}<asr_text>", .{l})));
    const audios = try cx.a.alloc(api.Audio, 1);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(features), &hash, .{});
    audios[0] = .{ .features = features, .frames = @intCast(frames), .at = @intCast(at), .tokens = @intCast(count), .hash = hash };
    const key = try cx.a.dupe(u32, ids.items);
    for (0..count) |i| key[at + i] = 0x80000000 | @as(u32, @truncate(std.hash.Wyhash.hash(i, &hash) & 0x7fffffff));
    return .{ .ids = ids.items, .audios = audios, .cache_key = key };
}

fn base64(cx: *Cx, text: []const u8) errors.Refused![]u8 {
    if (text.len > 36 * 1024 * 1024) return cx.refuse("audio upload exceeds 25 MiB");
    const decoder = std.base64.standard.Decoder;
    const n = decoder.calcSizeForSlice(text) catch return cx.refuse("invalid Base64 audio");
    if (n > 25 * 1024 * 1024) return cx.refuse("audio upload exceeds 25 MiB");
    const bytes = try cx.a.alloc(u8, n);
    decoder.decode(bytes, text) catch return cx.refuse("invalid Base64 audio");
    return bytes;
}

fn urlBytes(srv: *Server, cx: *Cx, url: []const u8) errors.Refused![]const u8 {
    if (std.mem.startsWith(u8, url, "data:audio/")) {
        const comma = std.mem.indexOf(u8, url, ";base64,") orelse return cx.refuse("audio data URL requires Base64 encoding");
        return base64(cx, url[comma + 8 ..]);
    }
    const fetch = @import("media_fetch.zig");
    var failure: fetch.Failure = .{};
    const types = [_][]const u8{ "audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/x-flac", "audio/ogg", "audio/mp4", "video/mp4", "video/webm", "audio/webm", "application/octet-stream" };
    const deadline = std.Io.Clock.awake.now(srv.io).addDuration(.fromSeconds(30));
    const body = fetch.fetch(cx.a, srv.io, url, 25 * 1024 * 1024, deadline, &types, &failure) catch return cx.fail(.request, "audio download failed: {s}", .{failure.text});
    return body.data;
}

pub fn render(srv: *Server, cx: *Cx, messages: json.Value) errors.Refused!Rendered {
    if (messages != .array or messages.array.len == 0) return cx.refuse("messages must be a non-empty list");
    var audio: ?[]const u8 = null;
    var context: std.ArrayList(u8) = .empty;
    var user_seen = false;
    var system_seen = false;
    for (messages.array) |message| {
        const role = message.strField("role") orelse return cx.refuse("message requires a role");
        const content = message.get("content") orelse return cx.refuse("message requires content");
        const system = std.mem.eql(u8, role, "system");
        if (!system and !std.mem.eql(u8, role, "user")) return cx.refuse("Qwen3-ASR accepts system and user messages");
        if (system) {
            if (system_seen or user_seen) return cx.refuse("Qwen3-ASR accepts one optional system context before its user turn");
            system_seen = true;
        } else {
            if (user_seen) return cx.refuse("Qwen3-ASR accepts one user turn");
            user_seen = true;
        }
        if (content == .string) {
            if (system) try context.appendSlice(cx.a, content.string);
            continue;
        }
        if (content != .array) return cx.refuse("audio message content must be a list");
        for (content.array) |part| {
            const kind = part.strField("type") orelse return cx.refuse("content part requires type");
            if (std.mem.eql(u8, kind, "text")) {
                const t = part.strField("text") orelse return cx.refuse("text part requires text");
                if (system) try context.appendSlice(cx.a, t);
            } else {
                if (system or audio != null) return cx.refuse("exactly one user audio part is supported");
                if (std.mem.eql(u8, kind, "audio_url")) {
                    const u = part.get("audio_url") orelse return cx.refuse("audio_url part requires audio_url");
                    audio = try urlBytes(srv, cx, if (u == .string) u.string else u.strField("url") orelse return cx.refuse("audio_url requires url"));
                } else if (std.mem.eql(u8, kind, "input_audio")) {
                    const u = part.get("input_audio") orelse return cx.refuse("input_audio part requires input_audio");
                    audio = try base64(cx, u.strField("data") orelse return cx.refuse("input_audio requires Base64 data"));
                } else return cx.refuse("Qwen3-ASR supports audio_url, input_audio and text parts");
            }
        }
    }
    const bytes = audio orelse return cx.refuse("Qwen3-ASR requires one audio part");
    const samples = core.audio_decode.decode(cx.a, bytes) catch |e| return cx.fail(.request, "audio decode failed: {s}", .{@errorName(e)});
    return prepare(srv, cx, samples, context.items, null, true);
}
