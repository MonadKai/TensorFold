const std = @import("std");
const cuda = @import("cuda");
const timing = @import("timing.zig");
const weights = @import("weights.zig");
const Kernels = @import("kernels.zig").Kernels;

pub const rows = 256;
pub const Probe = struct { ctx: *anyopaque, take: *const fn (*anyopaque, []const u8, u64, usize, usize, bool) anyerror!void };
pub const Scratch = struct { ids: u64, pos: u64, slots: u64, x: u64, norm: u64, q: u64, k: u64, v: u64, qr: u64, kr: u64, vr: u64, att: u64, tmp: u64, gate: u64, up: u64, act: u64, logits: u64 };

pub const Model = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    w: *const weights.Weights,
    k: *Kernels,
    context: usize,
    slots: usize,
    probe: ?Probe = null,
    timing: ?timing.Timing = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    kc: [28]u64 = undefined,
    vc: [28]u64 = undefined,
    inv: u64 = 0,
    x: Scratch = undefined,

    pub fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, @max(256, bytes));
        errdefer b.free();
        try m.buffers.append(m.a, b);
        return b.ptr;
    }

    pub fn upload(m: *Model, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, m.s);
    }

    pub fn copy(m: *Model, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn tap(m: *Model, name: []const u8, ptr: u64, r: usize, width: usize, bf16: bool) !void {
        if (m.probe) |p| try p.take(p.ctx, name, ptr, r, width, bf16);
    }

    pub fn streamBytes(context: usize) usize {
        return 28 * 2 * context * 1024 * 2;
    }

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights.Weights, k: *Kernels, context: usize, slots: usize) !Model {
        var m: Model = .{ .a = a, .d = d, .s = s, .w = w, .k = k, .context = context, .slots = slots };
        errdefer m.deinit();
        for (&m.kc, &m.vc) |*kc, *vc| {
            kc.* = try m.alloc(slots * context * 1024 * 2);
            vc.* = try m.alloc(slots * context * 1024 * 2);
            const bytes = slots * context * 1024 * 2;
            try (cuda.DeviceBuffer{ .d = d, .ptr = kc.*, .len = bytes, .borrowed = true }).fill8(0, s.handle);
            try (cuda.DeviceBuffer{ .d = d, .ptr = vc.*, .len = bytes, .borrowed = true }).fill8(0, s.handle);
        }
        m.inv = try m.alloc(64 * 4);
        var inv: [64]f32 = undefined;
        for (&inv, 0..) |*v, i| v.* = 1 / std.math.pow(f32, 1000000, @as(f32, @floatFromInt(2 * i)) / 128);
        try m.upload(m.inv, std.mem.sliceAsBytes(&inv));
        m.x = .{ .ids = try m.alloc(rows * 4), .pos = try m.alloc(rows * 4), .slots = try m.alloc(rows * 4), .x = try m.alloc(rows * 2048 * 2), .norm = try m.alloc(rows * 2048 * 2), .q = try m.alloc(rows * 2048 * 2), .k = try m.alloc(rows * 1024 * 2), .v = try m.alloc(rows * 1024 * 2), .qr = try m.alloc(rows * 2048 * 2), .kr = try m.alloc(rows * 1024 * 2), .vr = try m.alloc(rows * 1024 * 2), .att = try m.alloc(rows * 2048 * 2), .tmp = try m.alloc(rows * 2048 * 2), .gate = try m.alloc(rows * 6144 * 2), .up = try m.alloc(rows * 6144 * 2), .act = try m.alloc(rows * 6144 * 2), .logits = try m.alloc(slots * w.c.vocab * 2) };
        try s.synchronize();
        m.timing = try timing.Timing.init(d);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.s.synchronize() catch {};
        if (m.timing) |*t| t.deinit();
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.a);
    }

    pub fn begin(m: *Model, stage: timing.Stage) !void {
        if (m.timing) |*t| try t.start(m.s, stage);
    }

    pub fn end(m: *Model, stage: timing.Stage) !void {
        if (m.timing) |*t| try t.end(m.s, stage);
    }

    pub fn charge(m: *Model, streams: []const *StreamType) !void {
        if (m.timing) |*t| {
            const sum = try t.resolve();
            for (streams) |stream| {
                if (stream.audio_time == null) stream.audio_time = .{};
                stream.audio_time.?.add(sum);
            }
        }
    }

    const StreamType = @import("lanes").Stream;

    pub fn forward(m: *Model, slot: usize, start: usize, ids: []const u32, audio: u64, audio_at: usize, audio_rows: usize, head: bool) !void {
        const n = ids.len;
        if (n == 0 or n > rows or slot >= m.slots or start + n > m.context) return error.BadRows;
        const k = m.k;
        const s = m.s;
        const x = m.x;
        try m.upload(x.ids, std.mem.sliceAsBytes(ids));
        var pos: [rows]i32 = undefined;
        for (pos[0..n], 0..) |*v, i| v.* = @intCast(start + i);
        try m.upload(x.pos, std.mem.sliceAsBytes(pos[0..n]));
        try m.begin(.prefill);
        try k.embed.bf16(s, x.ids, m.w.embed, x.x, n, 2048);
        const lo = @max(start, audio_at);
        const hi = @min(start + n, audio_at + audio_rows);
        if (audio != 0 and hi > lo) try m.copy(x.x + (lo - start) * 4096, audio + (lo - audio_at) * 4096, (hi - lo) * 4096);
        try m.tap("decoder.embeddings", x.x, n, 2048, true);
        try m.decoder(n, slot, start, false);
        try m.end(.prefill);
        if (head) try m.projectHead(x.x + (n - 1) * 4096, 1);
    }

    pub fn decode(m: *Model, ids: []const u32, positions: []const i32, slots: []const i32) !void {
        const n = ids.len;
        if (n == 0 or n > m.slots or positions.len != n or slots.len != n) return error.BadRows;
        var occupied: [8]bool = @splat(false);
        for (positions, slots) |position, slot| {
            if (slot < 0 or slot >= m.slots or position < 0 or position >= m.context) return error.BadRows;
            if (occupied[@intCast(slot)]) return error.DuplicateSlot;
            occupied[@intCast(slot)] = true;
        }
        const x = m.x;
        try m.upload(x.ids, std.mem.sliceAsBytes(ids));
        try m.upload(x.pos, std.mem.sliceAsBytes(positions));
        try m.upload(x.slots, std.mem.sliceAsBytes(slots));
        if (m.timing) |*t| t.rows = n;
        try m.begin(.decode);
        try m.k.embed.bf16(m.s, x.ids, m.w.embed, x.x, n, 2048);
        try m.decoder(n, 0, 0, true);
        try m.end(.decode);
        try m.projectHead(x.x, n);
    }

    fn decoder(m: *Model, n: usize, slot: usize, start: usize, batch: bool) !void {
        const k = m.k;
        const s = m.s;
        const x = m.x;
        const window: cuda.window_attention.Window = .{ .h = 16, .hk = 8, .d = 128, .win = m.context, .ring = m.context, .scale = 1 / @sqrt(@as(f32, 128)) };
        for (&m.w.dec, 0..) |*l, i| {
            try k.norms.addRmsH(s, x.x, null, l.norm, .bf16, x.x, x.norm, null, n, 2048, 1e-6);
            try k.gemms.run(s, x.norm, l.q, x.q, n, 2048, 2048, .bf16);
            try k.gemms.run(s, x.norm, l.k, x.k, n, 1024, 2048, .bf16);
            try k.gemms.run(s, x.norm, l.v, x.v, n, 1024, 2048, .bf16);
            const kc = m.kc[i] + (if (batch) 0 else slot * m.context * 2048);
            const vc = m.vc[i] + (if (batch) 0 else slot * m.context * 2048);
            try k.qk.rowsFrom(s, .{ .q = x.q, .q_ld = 2048, .q_hs = 128, .k = x.k, .k_ld = 1024, .k_hs = 128, .v = x.v, .v_ld = 1024, .v_hs = 128 }, .{ .h = 16, .hk = 8, .dim = 128 }, .{ .norms = l.qnorm, .norm_k = l.knorm, .norms_bf16 = true, .inv = m.inv, .rot = 128 }, x.pos, if (batch) x.slots else null, .{ .q = x.qr, .k = x.kr, .v = x.vr, .kc = kc, .vc = vc, .len = m.context }, n);
            if (batch or n == 1) try k.attn.decode(s, x.qr, kc, vc, x.att, x.pos, if (batch) x.slots else null, window, 0, n) else try k.attn.prompt(s, x.qr, kc, vc, x.att, start, n, window);
            try k.gemms.run(s, x.att, l.o, x.tmp, n, 2048, 2048, .bf16);
            try k.vit.pointwise(s, x.tmp, x.x, x.x, n * 2048, .none);
            try k.norms.addRmsH(s, x.x, null, l.post, .bf16, x.x, x.norm, null, n, 2048, 1e-6);
            try k.gemms.run(s, x.norm, l.gate, x.gate, n, 6144, 2048, .bf16);
            try k.gemms.run(s, x.norm, l.up, x.up, n, 6144, 2048, .bf16);
            try k.act.swigluRound(s, x.gate, 6144, x.up, 6144, x.act, null, n, 6144);
            try k.gemms.run(s, x.act, l.down, x.tmp, n, 2048, 6144, .bf16);
            try k.vit.pointwise(s, x.tmp, x.x, x.x, n * 2048, .none);
            if (i == 0 or i == 27) {
                var label: [64]u8 = undefined;
                try m.tap(try std.fmt.bufPrint(&label, "decoder.layer{d}", .{i}), x.x, n, 2048, true);
            }
        }
    }

    fn projectHead(m: *Model, input: u64, n: usize) !void {
        try m.begin(.head);
        try m.k.norms.addRmsH(m.s, input, null, m.w.norm, .bf16, input, m.x.norm, null, n, 2048, 1e-6);
        try m.k.gemms.run(m.s, m.x.norm, m.w.head, m.x.logits, n, m.w.c.vocab, 2048, .bf16);
        try m.tap("decoder.logits", m.x.logits, n, m.w.c.vocab, true);
        try m.end(.head);
    }
};
