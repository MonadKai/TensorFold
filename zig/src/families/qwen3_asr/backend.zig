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
    resident_rules: u64,
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
        return .{ .m = m, .enc = try Encoder.init(m), .free = free, .lasts = lasts, .audio_buffers = audio, .result = try m.alloc(256), .rule = try m.alloc(256), .resident_rules = try m.alloc(m.slots * 32), .meta = try m.alloc(256) };
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
        return .{ .copy_drafts = false, .exact_width = 1, .plain_streams = true, .hidden_rows = true, .gpu_tokens = false, .mtp = false, .speculate = false, .drafts = 0, .max_streams = @intCast(b.m.slots), .batch_rows = @intCast(b.m.slots) };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = step, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    fn self(p: *anyopaque) *Backend {
        return @ptrCast(@alignCast(p));
    }

    fn open(b: *Backend, s: *Stream) !*Lane {
        if (b.active.getPtr(s)) |l| return l;
        errdefer b.m.discardTiming();
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
        var rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(s.sampling orelse .{ .seed = 0, .top_k = 1, .top_p = 1 }));
        try b.m.upload(b.resident_rules + slot * 32, std.mem.asBytes(&rule));
        const got = try b.active.getOrPut(b.m.a, s);
        b.free[slot] = false;
        got.value_ptr.* = .{ .slot = slot, .audio = b.audio_buffers[slot], .audio_at = au.at, .audio_rows = au.tokens, .last = b.lasts[slot] };
        return got.value_ptr;
    }

    fn step(p: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
        const b = self(p);
        errdefer b.m.discardTiming();
        const l = try b.open(s);
        if (s.isCancelled()) return error.Cancelled;
        const budget = if (max_rows == 0) model.rows else @min(model.rows, max_rows);
        const end = @min(s.prompt_len, l.at + budget);
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
        errdefer b.m.discardTiming();
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
        const pos = [_]i32{@intCast(l.len)};
        const slots = [_]i32{@intCast(l.slot)};
        try b.m.decode(&.{token}, &pos, &slots);
        l.len += 1;
        var picked: [1]u32 = undefined;
        try b.drawBatch(&.{s}, &.{position}, b.m.x.logits, &picked);
        const handle = b.next;
        b.next += 1;
        b.ring[handle % b.ring.len] = picked[0];
        return handle;
    }

    fn drawBatch(b: *Backend, streams: []const *Stream, positions: []const u64, logits: u64, out: []u32) !void {
        errdefer b.m.discardTiming();
        const n = streams.len;
        var meta: [8]i32 = undefined;
        var sampled = false;
        for (streams, positions, 0..) |s, position, i| {
            const l = b.active.getPtr(s) orelse return error.UnknownStream;
            meta[i] = @intCast(position - 1);
            sampled = sampled or s.sampling != null;
            try b.m.copy(b.rule + i * 32, b.resident_rules + l.slot * 32, 32);
        }
        try b.m.upload(b.meta, std.mem.sliceAsBytes(meta[0..n]));
        try b.m.begin(.sample);
        try b.m.k.round.argmax(b.m.s, logits, b.result, n, b.m.w.c.vocab, b.m.w.c.vocab);
        if (sampled) try b.m.k.draws.drawRows(b.m.s, logits, b.m.w.c.vocab, b.rule, b.meta, b.result + n * 8, n, null);
        try b.m.end(.sample);
        try b.m.s.synchronize();
        try b.m.charge(streams);
        var result: [24]u32 = undefined;
        try b.m.d.check(b.m.d.api.cuMemcpyDtoH_v2(@ptrCast(&result), b.result, n * (if (sampled) @as(usize, 12) else 8)), "batch picks");
        for (streams, out, 0..) |s, *value, i| {
            if (s.sampling == null and result[i * 2 + 1] != 0) return error.NonFiniteLogits;
            value.* = if (s.sampling == null) result[i * 2] else result[2 * n + i];
            if (value.* >= b.m.w.c.vocab) return error.BadToken;
        }
    }

    fn read(p: *anyopaque, handle: u64) anyerror!u32 {
        const b = self(p);
        if (handle >= b.next or b.next - handle > b.ring.len) return error.BadHandle;
        return b.ring[handle % b.ring.len];
    }

    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(p);
        const n = windows.len;
        if (n == 0) return;
        if (n > b.m.slots or out.len != n) return error.BadRows;
        var streams: [8]*Stream = undefined;
        var ids: [8]u32 = undefined;
        var slots: [8]i32 = undefined;
        var positions: [8]i32 = undefined;
        var draws: [8]u64 = undefined;
        var picked: [8]u32 = undefined;
        for (windows, 0..) |w, i| {
            if (w.rows() != 1 or w.masks.len != 0 or out[i].sampled.len == 0) return error.DraftsUnsupported;
            const l = b.active.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.positions[0] != l.len + 1) return error.PositionMismatch;
            streams[i] = w.stream;
            ids[i] = w.pending;
            slots[i] = @intCast(l.slot);
            positions[i] = @intCast(l.len);
            draws[i] = w.positions[0];
        }
        try b.m.decode(ids[0..n], positions[0..n], slots[0..n]);
        try b.drawBatch(streams[0..n], draws[0..n], b.m.x.logits, picked[0..n]);
        for (streams[0..n], out, picked[0..n]) |stream, o, token| {
            b.active.getPtr(stream).?.len += 1;
            o.sampled[0] = token;
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
        b.m.discardTiming();
        if (b.active.fetchRemove(s)) |entry| b.free[entry.value.slot] = true;
    }
};
