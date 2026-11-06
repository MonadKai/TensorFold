//! Host-only refusal checks never initialize CUDA or dereference launch state.
const std = @import("std");
const cuda = @import("root.zig");

test "shared row GEMM rejects malformed rows and pointers before launch" {
    const shared: cuda.row_gemm.Shared = .{ .f = undefined };
    for ([_]usize{ 0, 9, std.math.maxInt(usize) }) |rows|
        try std.testing.expectError(error.BadRows, shared.run(undefined, 1, 2, 3, rows, 4096, 2048));
    for ([_][3]u64{ .{ 0, 2, 3 }, .{ 1, 0, 3 }, .{ 1, 2, 0 } }) |pointers|
        try std.testing.expectError(error.BadRows, shared.run(undefined, pointers[0], pointers[1], pointers[2], 1, 4096, 2048));
}

test "shared row GEMM rejects unsupported geometry before launch" {
    const shared: cuda.row_gemm.Shared = .{ .f = undefined };
    for ([_][2]usize{ .{ 0, 2048 }, .{ 4096, 0 }, .{ 4097, 2048 }, .{ 2048, 6145 } }) |shape|
        try std.testing.expectError(error.UnsupportedRowGemmShape, shared.run(undefined, 1, 2, 3, 8, shape[0], shape[1]));
}

fn oldVersion() callconv(.c) usize {
    return 130000;
}

test "shared row GEMM rejects unqualified BLAS before querying CUDA" {
    var library: cuda.cublaslt.Library = undefined;
    library.api.cublasLtGetVersion = oldVersion;
    var driver: cuda.Driver = undefined;
    try std.testing.expectError(error.UnqualifiedRowGemmRuntime, cuda.row_gemm.Shared.init(&driver, &library, undefined));
}
