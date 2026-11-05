//! Actual checkpoint projections, rotating decoder layers to avoid a single-weight L2 benchmark.
const std = @import("std");
const cuda = @import("cuda");
const Weights = @import("weights.zig").Weights;
const Linear = cuda.cublaslt.Linear;
const mib = 1024 * 1024;
const Shape = struct { name: []const u8, n: usize, k: usize };
const shapes = [_]Shape{
    .{ .name = "qkv", .n = 4096, .k = 2048 },
    .{ .name = "attention_output", .n = 2048, .k = 2048 },
    .{ .name = "gate_up", .n = 12288, .k = 2048 },
    .{ .name = "down", .n = 2048, .k = 6144 },
    .{ .name = "lm_head", .n = 151936, .k = 2048 },
};

const Rig = struct {
    a: std.mem.Allocator,
    s: cuda.Stream,
    w: *const Weights,
    x: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    workspace: cuda.DeviceBuffer,
    begin: cuda.Event,
    end: cuda.Event,
    stable: *cuda.vit.Gemms,
    staged: ?*cuda.row_gemm.Rows = null,

    fn weight(r: *const Rig, shape: usize, layer: usize) u64 {
        const d = r.w.dec[layer % r.w.dec.len];
        return switch (shape) {
            0 => d.qkv,
            1 => d.o,
            2 => d.gate_up,
            3 => d.down,
            4 => r.w.head,
            else => unreachable,
        };
    }

    fn run(r: *Rig, shape: usize, rows: usize, plan: ?*const Linear, iteration: usize) !void {
        const f = shapes[shape];
        const w = r.weight(shape, iteration);
        if (r.staged) |p| {
            try p.run(r.s, r.x.ptr, w, r.y.ptr, rows, f.n, f.k);
        } else if (plan) |p| {
            try p.run(r.x.ptr, w, r.y.ptr, r.workspace.ptr, r.workspace.len, r.s.handle);
        } else {
            try r.stable.runRowStable(r.s, r.x.ptr, w, r.y.ptr, rows, f.n, f.k, .bf16);
        }
    }

    fn measure(r: *Rig, shape: usize, rows: usize, plan: ?*const Linear) ![7]f64 {
        for (0..28) |i| try r.run(shape, rows, plan, i);
        try r.s.synchronize();
        var samples: [7]f64 = undefined;
        for (&samples) |*sample| {
            try r.begin.record(r.s);
            for (0..28) |i| try r.run(shape, rows, plan, i);
            try r.end.record(r.s);
            try r.end.synchronize();
            sample.* = @as(f64, try cuda.Event.elapsedMs(r.begin, r.end)) / 28;
        }
        return samples;
    }

    fn emit(r: *Rig, value: anytype) !void {
        const text = try std.json.Stringify.valueAlloc(r.a, value, .{});
        defer r.a.free(text);
        std.debug.print("ASR_GEMM {s}\n", .{text});
    }
};

fn median(samples: [7]f64) f64 {
    var sorted = samples;
    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
    return sorted[3];
}

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and args.len != 3 and args.len != 8) {
        std.debug.print("usage: tf-qwen3-asr-gemm-bench MODEL [--staged | --trace stable|dense|staged ROWS SHAPE CANDIDATE WORKSPACE_MIB]\n", .{});
        return 2;
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var s = try cuda.Stream.init(&d, false);
    defer s.deinit();
    var w = try @import("weights.zig").load(a, init.io, &d, args[1]);
    defer w.deinit();
    var stable = try cuda.vit.Gemms.init(a, &d);
    defer stable.deinit();
    var x = try cuda.DeviceBuffer.alloc(&d, 32 * 6144 * 2);
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(&d, 32 * 151936 * 2);
    defer y.free();
    var workspace = try cuda.DeviceBuffer.alloc(&d, 64 * mib);
    defer workspace.free();
    var begin = try cuda.Event.init(&d, true);
    defer begin.deinit();
    var end = try cuda.Event.init(&d, true);
    defer end.deinit();
    const host = try a.alloc(u16, x.len / 2);
    defer a.free(host);
    var random = std.Random.DefaultPrng.init(20261106);
    for (host) |*v| {
        const f = random.random().float(f32) * 2 - 1;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    try x.upload(0, std.mem.sliceAsBytes(host));
    var r: Rig = .{ .a = a, .s = s, .w = &w, .x = x, .y = y, .workspace = workspace, .begin = begin, .end = end, .stable = &stable };
    try r.emit(.{ .schema = 1, .cublaslt_version = stable.lt.api.cublasLtGetVersion(), .seed = 20261106, .layers_per_sample = 28, .samples = 7, .rotating_decoder_weights = true, .includes_staging = false });
    if (args.len == 3) {
        if (!std.mem.eql(u8, args[2], "--staged")) return 2;
        var staged = try cuda.row_gemm.Rows.init(&d, &stable.lt);
        defer staged.deinit();
        r.staged = &staged;
        for (shapes, 0..) |f, shape| for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            const times = try r.measure(shape, rows, null);
            try r.emit(.{ .shape = f.name, .n = f.n, .k = f.k, .rows = rows, .physical_rows = 8, .mode = "staged", .milliseconds = times, .median_ms = median(times), .configuration = try staged.plans[shape].configuration() });
        };
        return 0;
    }
    if (args.len == 8) {
        if (!std.mem.eql(u8, args[2], "--trace")) return 2;
        const rows = try std.fmt.parseInt(usize, args[4], 10);
        const shape = try std.fmt.parseInt(usize, args[5], 10);
        const candidate = try std.fmt.parseInt(usize, args[6], 10);
        const limit = try std.fmt.parseInt(usize, args[7], 10) * mib;
        if (shape >= shapes.len or rows == 0 or rows > 32 or limit > workspace.len) return 2;
        if (std.mem.eql(u8, args[3], "stable")) {
            if (rows > 8) return 2;
            try r.run(shape, rows, null, 0);
        } else if (std.mem.eql(u8, args[3], "staged")) {
            if (rows > 8) return 2;
            var staged = try cuda.row_gemm.Rows.init(&d, &stable.lt);
            defer staged.deinit();
            r.staged = &staged;
            try r.run(shape, rows, null, 0);
            try s.synchronize();
        } else if (std.mem.eql(u8, args[3], "dense")) {
            const f = shapes[shape];
            var p = try Linear.init(&stable.lt, rows, f.n, f.k, .bf16, limit);
            defer p.deinit();
            var found: [32]cuda.cublaslt.Heuristic = undefined;
            const candidates = try p.candidates(&found);
            if (candidate >= candidates.len) return 2;
            try p.useAlgorithm(candidates[candidate].algo, limit);
            try r.emit(.{ .configuration = try p.configuration(), .numerical_flags = try p.numericalFlags(), .workspace_need = p.workspace_need });
            try r.run(shape, rows, &p, 0);
        } else return 2;
        try s.synchronize();
        return 0;
    }
    for (shapes, 0..) |f, shape| {
        for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            const times = try r.measure(shape, rows, null);
            try r.emit(.{ .shape = f.name, .n = f.n, .k = f.k, .rows = rows, .mode = "stable_before", .milliseconds = times, .median_ms = median(times) });
        }
        for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |rows| {
            var seen: std.AutoHashMapUnmanaged([8]u64, void) = .empty;
            defer seen.deinit(a);
            for ([_]usize{ 0, 8, 32, 64 }) |workspace_mib| {
                const limit = workspace_mib * mib;
                var p = Linear.init(&stable.lt, rows, f.n, f.k, .bf16, limit) catch |e| {
                    try r.emit(.{ .shape = f.name, .rows = rows, .workspace_mib = workspace_mib, .failed = @errorName(e) });
                    continue;
                };
                defer p.deinit();
                var found: [32]cuda.cublaslt.Heuristic = undefined;
                for (try p.candidates(&found), 0..) |candidate, index| {
                    if (candidate.state != 0) continue;
                    const entry = try seen.getOrPut(a, candidate.algo.data);
                    if (entry.found_existing) continue;
                    try p.useAlgorithm(candidate.algo, limit);
                    const split = try p.algorithmConfig(2);
                    const flags = try p.numericalFlags();
                    if (split > 1) {
                        try r.emit(.{ .shape = f.name, .rows = rows, .workspace_mib = workspace_mib, .candidate = index, .split_k = split, .numerical_flags = flags, .excluded = "split_k" });
                        continue;
                    }
                    const times = try r.measure(shape, rows, &p);
                    try r.emit(.{ .shape = f.name, .n = f.n, .k = f.k, .rows = rows, .mode = "dense", .workspace_mib = workspace_mib, .workspace_need = p.workspace_need, .candidate = index, .algorithm_id = try p.algorithmConfig(0), .tile = try p.algorithmConfig(1), .split_k = split, .reduction = try p.algorithmConfig(3), .numerical_flags = flags, .algorithm = p.algo.data, .milliseconds = times, .median_ms = median(times) });
                }
            }
        }
        for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            const times = try r.measure(shape, rows, null);
            try r.emit(.{ .shape = f.name, .n = f.n, .k = f.k, .rows = rows, .mode = "stable_after", .milliseconds = times, .median_ms = median(times) });
        }
    }
    return 0;
}
