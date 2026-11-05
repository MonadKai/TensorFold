const std = @import("std");
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedFatbinDirectory;
    try @import("contract_fatbins.zig").run(std.heap.c_allocator, init.io, args[1]);
}
