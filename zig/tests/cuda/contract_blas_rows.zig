//! Strided one-row vendor projections preserve the original row algorithm and output bits.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("qwen_bf16_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

fn depth(s: c.Shape) usize {
    return s.k + 1;
}

pub const Projection = struct {
    rig: Rig,
    gemms: *cuda.vit.Gemms,
    dense: ?*cuda.row_gemm.Rows = null,
    shared: ?*cuda.row_gemm.Shared = null,
    compare_rows: bool = true,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    xd: u64 = 0,
    wd: u64 = 0,
    out: u64 = 0,
    baseline: u64 = 0,

    pub fn deinit(p: *Projection) void {
        if (p.arena) |*a| a.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Projection);
    pub fn kernel(p: *Projection) g.Kernel {
        return .{ .spec = .{
            .name = if (p.shared != null) (if (p.compare_rows) "cuda ASR shared-weight GEMM stable oracle" else "cuda ASR shared-weight GEMM FP64 accuracy") else if (p.dense == null) "cuda ASR strided-batch cuBLASLt projection" else if (p.compare_rows) "cuda ASR fixed-batch dense row invariance" else "cuda ASR fixed-batch dense FP64 accuracy",
            .replaces = if (!p.compare_rows) null else if (p.dense != null) "same fixed-batch projection with one logical row" else "single-row cuBLASLt projections",
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 4 }, .{ .size = 32 }, .{ .size = 2048, .step = 8, .min = 8 } },
            .served = if (p.dense != null or p.shared != null) &.{ .{ .rows = 1, .n = 4096, .k = 2048 }, .{ .rows = 2, .n = 2048, .k = 2048 }, .{ .rows = 4, .n = 12288, .k = 2048 }, .{ .rows = 7, .n = 2048, .k = 6144 }, .{ .rows = 8, .n = 151936, .k = 2048 } } else &.{ .{ .rows = 3, .n = 1024, .k = 2048 }, .{ .rows = 8, .n = 2048, .k = 2048 }, .{ .rows = 2, .n = 6144, .k = 2048 }, .{ .rows = 5, .n = 2048, .k = 6144 }, .{ .rows = 7, .n = 4096, .k = 2048 }, .{ .rows = 8, .n = 12288, .k = 2048 }, .{ .rows = 2, .n = 151936, .k = 2048 } },
        }, .ctx = p, .vt = &vt };
    }

    pub fn prepare(p: *Projection, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 8 or s.n == 0 or s.k == 0 or s.n % 8 != 0 or s.k % 8 != 0) return false;
        if ((p.dense != null or p.shared != null) and cuda.row_gemm.shapeIndex(s.n, s.k) == null) return false;
        p.rig.reset();
        if (p.arena) |*a| a.deinit();
        p.arena = std.heap.ArenaAllocator.init(p.rig.gpu.gpa);
        const a = p.arena.?.allocator();
        p.shape = s;
        var random = std.Random.DefaultPrng.init(seed);
        const rng = random.random();
        p.x = try a.alloc(u16, s.rows * s.k);
        p.w = try a.alloc(u16, s.n * s.k);
        for (p.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (p.w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) {
            ref.stress(p.x);
            for (p.x) |*v| if (v.* == 0x3f80) {
                v.* = 0x3e80;
            };
            @memset(p.w, 0x3f80);
        }
        if (fill == .nonfinite) {
            @memset(p.w, 0x3f80);
            for (0..@min(s.rows, 3)) |row| p.x[row * s.k] = ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[row];
        }
        p.xd = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.wd = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.out = try p.rig.output(s.rows * s.n * 2);
        p.baseline = try p.rig.scratch(s.rows * s.n * 2);
        return true;
    }

    pub fn run(p: *Projection) !void {
        const s = p.shape;
        if (p.shared) |shared| {
            try shared.run(p.rig.s, p.xd, p.wd, p.out, s.rows, s.n, s.k);
        } else if (p.dense) |dense| {
            try dense.run(p.rig.s, p.xd, p.wd, p.out, s.rows, s.n, s.k);
        } else try p.gemms.runRowStable(p.rig.s, p.xd, p.wd, p.out, s.rows, s.n, s.k, .bf16);
        try p.rig.s.synchronize();
    }

    pub fn replaced(p: *Projection) ![]const u8 {
        const s = p.shape;
        for (0..s.rows) |row| {
            if (p.dense) |dense| {
                try dense.run(p.rig.s, p.xd + row * s.k * 2, p.wd, p.baseline + row * s.n * 2, 1, s.n, s.k);
            } else try p.gemms.run(p.rig.s, p.xd + row * s.k * 2, p.wd, p.baseline + row * s.n * 2, 1, s.n, s.k, .bf16);
        }
        return p.rig.bytes(p.baseline, s.rows * s.n * 2);
    }

    pub fn count(p: *Projection) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Projection, out: []f64, mag: []f64) !void {
        const s = p.shape;
        for (out, mag, 0..) |*v, *m, i| {
            const row = p.x[(i / s.n) * s.k ..][0..s.k];
            const weight = p.w[(i % s.n) * s.k ..][0..s.k];
            v.* = ref.dot(row, weight);
            m.* = 0;
            for (row, weight) |x, w| m.* += @abs(@as(f64, c.fromBf16(x)) * c.fromBf16(w));
        }
    }

    pub fn simulate(p: *Projection, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const terms = try p.arena.?.allocator().alloc(f64, s.k);
        for (out, 0..) |*v, i| {
            for (terms, 0..) |*term, j| term.* = @as(f64, c.fromBf16(p.x[(i / s.n) * s.k + j])) * c.fromBf16(p.w[(i % s.n) * s.k + j]);
            v.* = core.contract.rows.sumIn(acc, order, terms);
        }
    }
};
