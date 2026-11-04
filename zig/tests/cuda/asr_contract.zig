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
    const window = @import("contract_window.zig");
    var attention = try window.load(&driver);
    defer attention.unload();
    var prompts: [window.cfgs.len]window.Window = undefined;
    for (&prompts, window.cfgs) |*prompt, cfg| prompt.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .f = cfg, .g = &attention };
    defer for (&prompts) |*prompt| prompt.deinit();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1) {
        if (args.len != 2 or !std.mem.eql(u8, args[1], "--window-sums")) return 2;
        const verdict = try core.contract.gates.sums(a, prompts[0].kernel(), 0x6b63);
        return @intFromBool(verdict == .fail);
    }
    for (&prompts) |*prompt| try kernels.append(a, prompt.kernel());
    var tally: core.contract.gates.Tally = .{};
    try core.contract.gates.runAll(a, kernels.items, 0x6b63, &tally);
    std.debug.print("ASR FOCUSED: {d} passed, {d} failed; {d}/{d} controls red; full backend gates still required\n", .{ tally.passed, tally.failed, tally.caught, tally.controls });
    return @intFromBool(tally.problems() != 0);
}
