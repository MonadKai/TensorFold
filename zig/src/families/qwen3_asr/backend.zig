const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("model.zig");
const Encoder = @import("encoder.zig").Encoder;
const be = lanes.backend;
const Stream = lanes.Stream;

const Lane = struct { slot: usize, at: usize = 0, len: usize = 0, audio: u64 = 0, audio_at: usize = 0, audio_rows: usize = 0, last: u64 };

pub const Backend = struct {
    m: *model.Model,
    enc: Encoder,
    active: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool,
    lasts: []u64,
    audio_buffers: []u64,
    result: u64,
    rule: u64,
    meta: u64,
    ring: [1024]u32 = undefined,
    next: u64 = 0,

    pub fn init(m: *model.Model) !Backend {
        const free = try m.a.alloc(bool, m.slots);
        errdefer m.a.free(free);
        @memset(free, true);
        const lasts = try m.a.alloc(u64, m.slots);
        errdefer m.a.free(lasts);
        const audio = try m.a.alloc(u64, m.slots);
        errdefer m.a.free(audio);
        for (lasts, audio) |*last, *au| {
            last.* = try m.alloc(m.w.c.vocab * 2);
            au.* = try m.alloc(@import("encoder.zig").max_rows * 2048 * 2);
        }
        return .{ .m = m, .enc = try Encoder.init(m), .free = free, .lasts = lasts, .audio_buffers = audio, .result = try m.alloc(256), .rule = try m.alloc(256), .meta = try m.alloc(256) };
    }

    pub fn deinit(b: *Backend) void {
        b.active.deinit(b.m.a);
        b.m.a.free(b.free);
        b.m.a.free(b.lasts);
        b.m.a.free(b.audio_buffers);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }

    pub fn facts(b: *Backend) lanes.Model {
        return .{ .copy_drafts = false, .exact_width = 1, .gpu_tokens = false, .mtp = false, .speculate = false, .drafts = 0, .max_streams = @intCast(b.m.slots), .batch_rows = @intCast(b.m.slots) };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = step, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    fn self(p: *anyopaque) *Backend {
        return @ptrCast(@alignCast(p));
    }

    fn open(b: *Backend, s: *Stream) !*Lane {
        if (b.active.getPtr(s)) |l| return l;
        if (s.audios.len != 1) return error.OneAudioRequired;
        const slot = for (b.free, 0..) |f, i| {
            if (f) break i;
        } else return error.NoFreeSlot;
        const au = s.audios[0];
        if (au.tokens != @import("config.zig").tokens(au.frames) or au.at > s.prompt_len or au.tokens > s.prompt_len - au.at)
            return error.BadAudioPosition;
        try b.m.begin(.encoder);
        const encoded = try b.enc.run(b.m, au.features, au.frames);
        try b.m.copy(b.audio_buffers[slot], encoded, @as(usize, au.tokens) * 4096);
        try b.m.end(.encoder);
        const got = try b.active.getOrPut(b.m.a, s);
        b.free[slot] = false;
        got.value_ptr.* = .{ .slot = slot, .audio = b.audio_buffers[slot], .audio_at = au.at, .audio_rows = au.tokens, .last = b.lasts[slot] };
        return got.value_ptr;
    }

    fn step(p: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const b = self(p);
        const l = try b.open(s);
        if (s.isCancelled()) return error.Cancelled;
        const end = @min(s.prompt_len, l.at + model.rows);
        const done = end == s.prompt_len;
        try b.m.forward(l.slot, l.at, s.prompt()[l.at..end], l.audio, l.audio_at, l.audio_rows, done);
        l.at = end;
        l.len = end;
        if (done) try b.m.copy(l.last, b.m.x.logits, b.m.w.c.vocab * 2);
        try b.m.s.synchronize();
        try b.m.charge(&.{s});
        s.fill_at.store(@intCast(end), .monotonic);
        return done;
    }

    fn prefill(p: *anyopaque, s: *Stream) anyerror!void {
        while (!try step(p, s, 0)) {}
    }

    fn draw(b: *Backend, s: *Stream, logits: u64, position: u64) !u64 {
        var token: u32 = undefined;
        try b.m.begin(.sample);
        if (s.sampling) |sp| {
            var rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
            var meta = [_]u64{ position - 1, 0 };
            try b.m.upload(b.rule, std.mem.asBytes(&rule));
            try b.m.upload(b.meta, std.mem.sliceAsBytes(&meta));
            try b.m.k.draws.draw(b.m.s, logits, b.m.w.c.vocab, b.rule, b.meta, 0, b.result, 1, null, null);
            try b.m.end(.sample);
            try b.m.s.synchronize();
            try b.m.charge(&.{s});
            try b.m.d.check(b.m.d.api.cuMemcpyDtoH_v2(@ptrCast(&token), b.result, 4), "draw token");
        } else {
            try b.m.k.round.argmax(b.m.s, logits, b.result, 1, b.m.w.c.vocab, b.m.w.c.vocab);
            try b.m.end(.sample);
            try b.m.s.synchronize();
            try b.m.charge(&.{s});
            var pick: cuda.tree_round.Pick = undefined;
            try b.m.d.check(b.m.d.api.cuMemcpyDtoH_v2(@ptrCast(&pick), b.result, @sizeOf(@TypeOf(pick))), "argmax token");
            if (pick.nonfinite != 0) return error.NonFiniteLogits;
            token = pick.token;
        }
        if (token >= b.m.w.c.vocab) return error.BadToken;
        const handle = b.next;
        b.next += 1;
        b.ring[handle % b.ring.len] = token;
        return handle;
    }

    fn first(p: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(p);
        const l = b.active.getPtr(s) orelse return error.UnknownStream;
        if (l.at != s.prompt_len or position != l.len) return error.PositionMismatch;
        return b.draw(s, l.last, position);
    }

    fn queue(p: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(p);
        const l = b.active.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len + 1) return error.PositionMismatch;
        const token = switch (feed) {
            .value => |v| v,
            .handle => |h| try read(p, h),
        };
        try b.m.forward(l.slot, l.len, &.{token}, 0, 0, 0, true);
        l.len += 1;
        return b.draw(s, b.m.x.logits, position);
    }

    fn read(p: *anyopaque, handle: u64) anyerror!u32 {
        const b = self(p);
        if (handle >= b.next or b.next - handle > b.ring.len) return error.BadHandle;
        return b.ring[handle % b.ring.len];
    }

    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows, out) |w, o| {
            if (w.rows() != 1 or w.masks.len != 0) return error.DraftsUnsupported;
            o.sampled[0] = try read(p, try queue(p, w.stream, .{ .value = w.pending }, w.positions[0]));
        }
    }

    fn keep(_: *anyopaque, _: []const be.Window, paths: []const []const u32) anyerror!void {
        for (paths) |path| if (path.len != 1 or path[0] != 0) return error.DraftsUnsupported;
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0) return error.DraftsUnsupported;
    }

    fn release(p: *anyopaque, s: *Stream) void {
        const b = self(p);
        if (b.active.fetchRemove(s)) |entry| b.free[entry.value.slot] = true;
    }
};
