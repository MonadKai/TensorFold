const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Attention = struct {
    const vtable = g.hooks(Attention);
    rig: rig.Rig,
    f: cuda.Function,
    fault: bool = false,
    source: []f32 = &.{},
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    input: u64 = 0,
    out: u64 = 0,

    pub fn deinit(t: *Attention) void {
        t.rig.gpu.gpa.free(t.source);
        t.rig.deinit();
    }

    fn depth(_: c.Shape) usize {
        return 512;
    }

    pub fn kernel(t: *Attention) g.Kernel {
        return .{ .ctx = t, .vt = &vtable, .control = if (t.fault) .shapes else null, .spec = .{
            .name = if (t.fault) "CONTROL ASR attention wrong window" else "cuda ASR bidirectional window attention, 16 heads of 64",
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .ops = 8,
            .served = &.{ .{ .rows = 390, .n = 1024, .k = 104 }, .{ .rows = 105, .n = 1024, .k = 104 }, .{ .rows = 13, .n = 1024, .k = 104 } },
            .tiles = .{ .{ .size = 104 }, .{}, .{} },
            .sums = .{ .why = "zero QK gives four exact 1/4 weights; V cancellation stays exact in fp32" },
        } };
    }

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 1024 or shape.k == 0 or shape.k > 128) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.source);
        t.source = &.{};
        t.shape = shape;
        t.source = try a.alloc(f32, shape.rows * 1024 * 3);
        const bits = try a.alloc(u16, t.source.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.source) |*v| v.* = (prng.random().float(f32) - 0.5) * 2;
        if (fill == .stress) {
            if (shape.rows != 4) return error.InvalidStressShape;
            @memset(t.source[0 .. 2 * shape.rows * 1024], 0);
            for (t.source[2 * shape.rows * 1024 ..], 0..) |*v, i|
                v.* = ([_]f32{ 8192, 1, -8192, 1 })[i / 1024];
        }
        if (fill == .nonfinite) {
            t.source[0] = std.math.nan(f32);
            t.source[shape.rows * 1024 + 65] = std.math.inf(f32);
            t.source[2 * shape.rows * 1024 + 130] = -std.math.inf(f32);
        }
        for (t.source, bits) |*v, *b| {
            b.* = c.toBf16(v.*);
            v.* = c.fromBf16(b.*);
        }
        t.input = try t.rig.input(std.mem.sliceAsBytes(bits));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn count(t: *Attention) usize {
        return t.shape.rows * 1024;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (t.shape.rows != 4) return error.InvalidStressShape;
        const value = core.contract.rows.sumIn(acc, order, &.{ 2048, 0.25, -2048, 0.25 });
        @memset(out, c.fromBf16(c.toBf16(@floatCast(value))));
    }

    pub fn run(t: *Attention) !void {
        const step = t.count() * 2;
        var args: cuda.Args = .{};
        for ([_]u64{ t.input, t.input + step, t.input + 2 * step, t.out }) |v| args.add(v);
        args.add(@as(i32, @intCast(t.shape.rows)));
        args.add(@as(i32, @intCast(if (t.fault) 1 else t.shape.k)));
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.shape.rows), .y = 16 }, .block = .{ .x = 128 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const size = t.count();
        const q = t.source[0..size];
        const k = t.source[size..][0..size];
        const v = t.source[2 * size ..];
        for (0..t.shape.rows) |row| {
            const first = row / t.shape.k * t.shape.k;
            const end = @min(first + t.shape.k, t.shape.rows);
            for (0..16) |head| {
                var scores: [128]f64 = undefined;
                var maximum: f64 = -std.math.inf(f64);
                for (first..end, 0..) |key, j| {
                    var dot: f64 = 0;
                    for (0..64) |d| dot += @as(f64, q[row * 1024 + head * 64 + d]) * k[key * 1024 + head * 64 + d];
                    scores[j] = dot * 0.125;
                    maximum = if (std.math.isNan(scores[j]) or std.math.isNan(maximum)) std.math.nan(f64) else @max(maximum, scores[j]);
                }
                var sum: f64 = 0;
                for (scores[0 .. end - first]) |*score| {
                    score.* = @exp(score.* - maximum);
                    sum += score.*;
                }
                for (0..64) |d| {
                    var result: f64 = 0;
                    var magnitude: f64 = 0;
                    for (first..end, scores[0 .. end - first]) |key, score| {
                        const term = score / sum * v[key * 1024 + head * 64 + d];
                        result += term;
                        magnitude += @abs(term);
                    }
                    const index = row * 1024 + head * 64 + d;
                    want[index] = result;
                    mag[index] = magnitude;
                }
            }
        }
    }
};
