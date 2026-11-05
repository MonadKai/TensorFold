//! A fixed physical batch keeps each logical row's arithmetic independent of scheduler composition.
const std = @import("std");
const cuda = @import("root.zig");
const blas = @import("cublaslt.zig");

pub const physical_rows = 32;
pub const logical_rows = 8;
pub const workspace_bytes = 32 * 1024 * 1024;
pub const policy_version = "asr-sm121-cublaslt130100-p32-v2";
pub const Shape = struct { n: usize, k: usize, workspace_limit: usize, numerical_flags: u64, configuration: [7]i32 };
pub const shapes = [_]Shape{
    .{ .n = 4096, .k = 2048, .workspace_limit = 8 * 1024 * 1024, .numerical_flags = 0x30202, .configuration = .{ 67, 13, 1, 0, 0, 28, 35 } },
    .{ .n = 2048, .k = 2048, .workspace_limit = 0, .numerical_flags = 0x30202, .configuration = .{ 67, 13, 1, 0, 0, 28, 35 } },
    .{ .n = 12288, .k = 2048, .workspace_limit = 0, .numerical_flags = 0x20202, .configuration = .{ 21, 11, 1, 0, 0, 0, 19 } },
    .{ .n = 2048, .k = 6144, .workspace_limit = workspace_bytes, .numerical_flags = 0x30202, .configuration = .{ 67, 13, 1, 0, 0, 19, 35 } },
    .{ .n = 151936, .k = 2048, .workspace_limit = 0, .numerical_flags = 0x20202, .configuration = .{ 21, 11, 1, 0, 0, 0, 19 } },
};

pub fn shapeIndex(n: usize, k: usize) ?usize {
    for (shapes, 0..) |shape, i| if (shape.n == n and shape.k == k) return i;
    return null;
}

/// One model stream owns plans, staging and workspace; queued calls on that stream cannot overlap them.
pub const Rows = struct {
    x: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    workspace: cuda.DeviceBuffer,
    plans: [shapes.len]blas.Linear = undefined,
    initialized: usize = 0,

    pub fn init(d: *const cuda.Driver, lt: *const blas.Library) !Rows {
        if (lt.api.cublasLtGetVersion() != 130100) return error.UnqualifiedRowGemmRuntime;
        if (try d.currentCapability() != 121) return error.UnqualifiedRowGemmRuntime;
        var x = try cuda.DeviceBuffer.alloc(d, physical_rows * 6144 * 2);
        errdefer x.free();
        var y = try cuda.DeviceBuffer.alloc(d, physical_rows * 151936 * 2);
        errdefer y.free();
        var workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        errdefer workspace.free();
        var self: Rows = .{ .x = x, .y = y, .workspace = workspace };
        errdefer for (self.plans[0..self.initialized]) |*plan| plan.deinit();
        for (shapes, &self.plans) |shape, *plan| {
            plan.* = try blas.Linear.init(lt, physical_rows, shape.n, shape.k, .bf16, shape.workspace_limit);
            self.initialized += 1;
            var found: [32]blas.Heuristic = undefined;
            var matched = false;
            for (try plan.candidates(&found)) |candidate| {
                if (candidate.state != 0) continue;
                try plan.useAlgorithm(candidate.algo, shape.workspace_limit);
                if (std.mem.eql(i32, &(try plan.configuration()), &shape.configuration)) {
                    if (try plan.numericalFlags() != shape.numerical_flags) return error.UnqualifiedRowGemmRuntime;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                std.log.err("row GEMM policy has no matching algorithm for N={d}, K={d}", .{ shape.n, shape.k });
                return error.UnqualifiedRowGemmAlgorithm;
            }
        }
        return self;
    }

    pub fn deinit(self: *Rows) void {
        for (self.plans[0..self.initialized]) |*plan| plan.deinit();
        self.workspace.free();
        self.y.free();
        self.x.free();
        self.* = undefined;
    }

    /// Only logical rows leave staging; inactive rows never acquire slots, positions, KV or sampler state.
    pub fn run(self: *Rows, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or rows > logical_rows or x == 0 or w == 0 or y == 0) return error.BadRows;
        const index = shapeIndex(n, k) orelse return error.UnsupportedRowGemmShape;
        const bytes = rows * k * 2;
        const padding: cuda.DeviceBuffer = .{ .d = self.x.d, .ptr = self.x.ptr + bytes, .len = (physical_rows - rows) * k * 2, .borrowed = true };
        try self.x.copyFrom(0, x, bytes, s.handle);
        try padding.fill8(0, s.handle);
        try self.plans[index].run(self.x.ptr, w, self.y.ptr, self.workspace.ptr, self.workspace.len, s.handle);
        const destination: cuda.DeviceBuffer = .{ .d = self.y.d, .ptr = y, .len = rows * n * 2, .borrowed = true };
        try destination.copyFrom(0, self.y.ptr, destination.len, s.handle);
    }
};
