const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const model = @import("model.zig");
const Kernels = @import("kernels.zig").Kernels;
const Encoder = @import("encoder.zig").Encoder;

const Record = struct { name: []const u8, file: []const u8, rows: usize, width: usize, dtype: []const u8 };
const Probe = struct {
    a: std.mem.Allocator,
    io: std.Io,
    m: *model.Model,
    dir: []const u8,
    records: std.ArrayList(Record) = .empty,

    fn take(p: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, bf16: bool) !void {
        const t: *Probe = @ptrCast(@alignCast(p));
        const filename = try std.fmt.allocPrint(t.a, "{d:0>4}-{s}.bin", .{ t.records.items.len, name });
        const data = try t.a.alloc(u8, rows * width * @as(usize, if (bf16) 2 else 4));
        defer t.a.free(data);
        try t.m.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = t.m.d, .ptr = ptr, .len = data.len };
        try b.download(0, data);
        const path = try std.fs.path.join(t.a, &.{ t.dir, filename });
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = data });
        try t.records.append(t.a, .{ .name = try t.a.dupe(u8, name), .file = filename, .rows = rows, .width = width, .dtype = if (bf16) "bf16" else "f32" });
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len != 5) {
        std.debug.print("usage: tf-qwen3-asr-check MODEL AUDIO CASE.json OUTDIR\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, argv[3], a, .limited(4 << 20));
    const spec = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const ids = try integers(a, spec.object.get("prompt_ids").?);
    const history = try integers(a, spec.object.get("teacher_tokens").?);
    const audio_at: usize = @intCast(spec.object.get("audio_at").?.integer);
    const audio_rows: usize = @intCast(spec.object.get("audio_tokens").?.integer);
    const file = try std.Io.Dir.cwd().readFileAlloc(io, argv[2], a, .limited(25 << 20));
    const samples = try core.audio_decode.decode(a, file);
    const plan = core.audio_features.Plan.init();
    const features = try plan.extract(a, samples);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var w = try @import("weights.zig").load(a, io, &driver, argv[1]);
    defer w.deinit();
    var k = try Kernels.load(a, &driver);
    defer k.deinit();
    var m = try model.Model.init(a, &driver, stream, &w, &k, 8192, 1);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(io, argv[4]);
    var probe: Probe = .{ .a = a, .io = io, .m = &m, .dir = argv[4] };
    m.probe = .{ .ctx = &probe, .take = Probe.take };
    var enc = try Encoder.init(&m);
    const audio = try enc.run(&m, features, samples.len / 160);
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + model.rows);
        try m.forward(0, at, ids[at..end], audio, audio_at, audio_rows, end == ids.len);
        at = end;
    }
    for (history, 0..) |token, i| try m.forward(0, ids.len + i, &.{token}, 0, 0, 0, true);
    const manifest = try std.json.Stringify.valueAlloc(a, .{ .records = probe.records.items, .prompt_ids = ids, .teacher_tokens = history }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ argv[4], "manifest.json" }), .data = manifest });
    return 0;
}

fn integers(a: std.mem.Allocator, value: std.json.Value) ![]u32 {
    const out = try a.alloc(u32, value.array.items.len);
    for (out, value.array.items) |*v, item| v.* = std.math.cast(u32, item.integer) orelse return error.BadToken;
    return out;
}
