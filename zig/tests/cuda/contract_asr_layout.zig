const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Kind = enum { mel, conv, pack };

pub const Layout = struct {
    const vtable = g.hooks(Layout);
    rig: rig.Rig,
    f: cuda.Function,
    kind: Kind,
    fault: bool = false,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    source: []f32 = &.{},
    input: u64 = 0,
    out: u64 = 0,

    pub fn deinit(t: *Layout) void {
        t.rig.gpu.gpa.free(t.source);
        t.rig.deinit();
    }

    pub fn kernel(t: *Layout) g.Kernel {
        return .{ .ctx = t, .vt = &vtable, .control = if (t.fault) .shapes else null, .spec = .{
            .name = switch (t.kind) {
                .mel => "cuda ASR mel im2col f32 to bf16",
                .conv => "cuda ASR convolution im2col",
                .pack => if (t.fault) "CONTROL ASR truncated convolution pack" else "cuda ASR convolution time-major pack",
            },
            .accum = .none,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 2 }, .{ .size = 2 }, .{} },
            .served = switch (t.kind) {
                .mel => &.{ .{ .rows = 128, .n = 100, .k = 1 }, .{ .rows = 128, .n = 5, .k = 1 } },
                .conv => &.{ .{ .rows = 64, .n = 50, .k = 480 }, .{ .rows = 32, .n = 25, .k = 480 } },
                .pack => &.{.{ .rows = 13, .n = 16, .k = 480 }},
            },
        } };
    }

    fn depth(_: c.Shape) usize {
        return 0;
    }

    pub fn prepare(t: *Layout, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or (t.kind == .mel and shape.k != 1)) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.source);
        t.source = &.{};
        t.shape = shape;
        t.source = try a.alloc(f32, shape.rows * shape.n * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.source) |*v| v.* = (prng.random().float(f32) - 0.5) * 4;
        if (fill == .nonfinite and t.source.len >= 4) {
            t.source[0] = std.math.nan(f32);
            t.source[t.source.len / 2] = std.math.inf(f32);
            t.source[t.source.len - 1] = -std.math.inf(f32);
        }
        if (t.kind == .mel) t.input = try t.rig.input(std.mem.sliceAsBytes(t.source)) else {
            const bits = try a.alloc(u16, t.source.len);
            defer a.free(bits);
            for (t.source, bits) |*v, *b| {
                b.* = c.toBf16(v.*);
                v.* = c.fromBf16(b.*);
            }
            t.input = try t.rig.input(std.mem.sliceAsBytes(bits));
        }
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn count(t: *Layout) usize {
        const s = t.shape;
        return if (t.kind == .pack) s.rows * s.n * s.k else ((s.rows + 1) / 2) * ((s.n + 1) / 2) * s.k * 9;
    }

    pub fn run(t: *Layout) !void {
        const s = t.shape;
        var args: cuda.Args = .{};
        args.add(t.input);
        args.add(t.out);
        if (t.kind == .pack) {
            for ([_]usize{ if (t.fault and s.rows > 1) s.rows - 1 else s.rows, s.n, s.k }) |v| args.add(@as(i32, @intCast(v)));
        } else {
            for ([_]usize{ s.rows, s.n, s.k, s.n, 0, @intFromBool(t.kind == .mel) }) |v| args.add(@as(i32, @intCast(v)));
        }
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast((t.count() + 255) / 256) }, .block = .{ .x = 256 } }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn reference(t: *Layout, want: []f64, mag: []f64) !void {
        const s = t.shape;
        for (want, mag, 0..) |*w, *m, i| {
            var value: f32 = 0;
            if (t.kind == .pack) {
                const time = i / (s.n * s.k);
                const channel = i / s.n % s.k;
                const frequency = i % s.n;
                value = t.source[(frequency * s.rows + time) * s.k + channel];
            } else {
                const col = i % (s.k * 9);
                const row = i / (s.k * 9);
                const y = @as(isize, @intCast(row / ((s.n + 1) / 2) * 2 + col / 3 % 3)) - 1;
                const x = @as(isize, @intCast(row % ((s.n + 1) / 2) * 2 + col % 3)) - 1;
                if (x >= 0 and y >= 0 and x < s.n and y < s.rows)
                    value = t.source[(@as(usize, @intCast(y)) * s.n + @as(usize, @intCast(x))) * s.k + col / 9];
            }
            w.* = c.fromBf16(c.toBf16(value));
            m.* = @abs(w.*);
        }
    }
};
