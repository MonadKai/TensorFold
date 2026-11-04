const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.c_allocator;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    const gpu: check.Gpu = .{ .gpa = a, .io = init.io, .d = &driver, .ctx = &context };
    var kernels: std.ArrayList(core.contract.gates.Kernel) = .empty;
    defer kernels.deinit(a);
    const group = try @import("contract_asr_group.zig").Group.create(a, gpu, stream, &kernels);
    defer group.destroy();
    var tally: core.contract.gates.Tally = .{};
    try core.contract.gates.runAll(a, kernels.items, 0x6b63, &tally);
    std.debug.print("ASR FOCUSED: {d} passed, {d} failed; {d}/{d} controls red; full backend gates still required\n", .{ tally.passed, tally.failed, tally.caught, tally.controls });
    return @intFromBool(tally.problems() != 0);
}
