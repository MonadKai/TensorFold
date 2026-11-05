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
    if (argv.len != 5 and !(argv.len == 6 and (std.mem.eql(u8, argv[5], "--cancel-profile") or std.mem.eql(u8, argv[5], "--batch-probe") or std.mem.eql(u8, argv[5], "--mixed-probe")))) {
        std.debug.print("usage: tf-qwen3-asr-check MODEL AUDIO CASE.json OUTDIR [--cancel-profile|--batch-probe|--mixed-probe]\n", .{});
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
    var m = try model.Model.init(a, &driver, stream, &w, &k, 8192, if (argv.len == 6 and !std.mem.eql(u8, argv[5], "--cancel-profile")) 8 else 1);
    defer m.deinit();
    if (argv.len == 6 and std.mem.eql(u8, argv[5], "--mixed-probe")) {
        try mixedProbe(a, io, &m, ids, features, samples.len / 160, audio_at, audio_rows, history, argv[4]);
        return 0;
    }
    if (argv.len == 6 and std.mem.eql(u8, argv[5], "--batch-probe")) {
        try batchProbe(a, io, &m, ids, features, samples.len / 160, audio_at, audio_rows, history, argv[4]);
        return 0;
    }
    if (argv.len == 6) {
        try cancelProfile(a, &m, ids, features, samples.len / 160, audio_at, audio_rows, history[0]);
        return 0;
    }
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
    for (history, 0..) |token, i| try m.decode(&.{token}, &.{@intCast(ids.len + i)}, &.{0});
    const manifest = try std.json.Stringify.valueAlloc(a, .{ .records = probe.records.items, .prompt_ids = ids, .teacher_tokens = history }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ argv[4], "manifest.json" }), .data = manifest });
    return 0;
}

fn integers(a: std.mem.Allocator, value: std.json.Value) ![]u32 {
    const out = try a.alloc(u32, value.array.items.len);
    for (out, value.array.items) |*v, item| v.* = std.math.cast(u32, item.integer) orelse return error.BadToken;
    return out;
}

fn cancelProfile(a: std.mem.Allocator, m: *model.Model, ids: []const u32, features: []const f32, frames: usize, audio_at: usize, audio_rows: usize, expected: u32) !void {
    const lanes = @import("lanes");
    if (m.timing == null) return error.ProfileRequired;
    var backend = try @import("backend.zig").Backend.init(m);
    defer backend.deinit();
    const b = backend.backend();
    const audios = [_]lanes.stream.Audio{.{ .features = features, .frames = @intCast(frames), .at = @intCast(audio_at), .tokens = @intCast(audio_rows), .hash = @splat(0) }};
    const spec: lanes.stream.Spec = .{ .id = "cancel-profile", .prompt = ids, .audios = &audios, .max_new = 1, .drafts = false };
    var cancelled = try lanes.Stream.init(a, spec);
    defer cancelled.deinit(a);
    const Always = struct {
        fn check(_: *anyopaque) bool {
            return true;
        }
    };
    cancelled.cancel_check = .{ .ptr = &cancelled, .check = Always.check };
    if (b.vtable.prefill_step.?(b.ptr, &cancelled, 0)) |_| return error.ExpectedCancellation else |e| if (e != error.Cancelled) return e;
    b.vtable.release(b.ptr, &cancelled);
    var recovered = try lanes.Stream.init(a, spec);
    defer recovered.deinit(a);
    defer b.vtable.release(b.ptr, &recovered);
    try b.prefill(&recovered);
    const token = try b.read(try b.first(&recovered, ids.len));
    if (token != expected) return error.RecoveryTokenMismatch;
    std.debug.print("PASS cancel after encoder: slot reused, timing resolved, recovery token {d} equals reference\n", .{token});
}

fn batchProbe(a: std.mem.Allocator, io: std.Io, m: *model.Model, ids: []const u32, features: []const f32, frames: usize, audio_at: usize, audio_rows: usize, history: []const u32, dir: []const u8) !void {
    var enc = try Encoder.init(m);
    const audio = try enc.run(m, features, frames);
    for (0..8) |slot| {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + model.rows);
            try m.forward(slot, at, ids[at..end], audio, audio_at, audio_rows, false);
            at = end;
        }
    }
    const Result = struct { step: usize, rows: usize, row: usize, different_columns: usize, top1_equal: bool };
    var records: std.ArrayList(Result) = .empty;
    const solo = try a.alloc(u16, m.w.c.vocab);
    const batch = try a.alloc(u16, m.w.c.vocab * 8);
    var failures: usize = 0;
    for (history, 0..) |token, step| {
        const pos = @as(i32, @intCast(ids.len + step));
        m.joined_projections = m.projections != null;
        try m.decode(&.{token}, &.{pos}, &.{0});
        try m.s.synchronize();
        try (cuda.DeviceBuffer{ .d = m.d, .ptr = m.x.logits, .len = solo.len * 2 }).download(0, std.mem.sliceAsBytes(solo));
        const positions: [8]i32 = @splat(pos);
        const tokens: [8]u32 = @splat(token);
        const slots = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
        m.joined_projections = true;
        for (2..9) |n| {
            try m.decode(tokens[0..n], positions[0..n], slots[0..n]);
            try m.s.synchronize();
            try (cuda.DeviceBuffer{ .d = m.d, .ptr = m.x.logits, .len = n * solo.len * 2 }).download(0, std.mem.sliceAsBytes(batch[0 .. n * solo.len]));
            for (0..n) |row| {
                const values = batch[row * solo.len ..][0..solo.len];
                var different: usize = 0;
                var best_solo: usize = 0;
                var best_batch: usize = 0;
                for (solo, values, 0..) |left, right, column| {
                    if (left != right) different += 1;
                    if (core.contract.spec.fromBf16(left) > core.contract.spec.fromBf16(solo[best_solo])) best_solo = column;
                    if (core.contract.spec.fromBf16(right) > core.contract.spec.fromBf16(values[best_batch])) best_batch = column;
                }
                if (different != 0) failures += 1;
                try records.append(a, .{ .step = step, .rows = n, .row = row, .different_columns = different, .top1_equal = best_solo == best_batch });
            }
        }
    }
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const output = try std.json.Stringify.valueAlloc(a, .{ .records = records.items, .failures = failures }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "batch-probe.json" }), .data = output });
    std.debug.print("{s} row-invariance: {d}/{d} teacher-forced logits rows differ between solo and batches 2..8\n", .{ if (failures == 0) "PASS" else "FAIL", failures, records.items.len });
    if (failures != 0) return error.RowArithmeticChanged;
}

fn prepareSlot(m: *model.Model, slot: usize, ids: []const u32, audio: u64, at: usize, count: usize) !void {
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(ids.len, start + model.rows);
        try m.forward(slot, start, ids[start..end], audio, at, count, false);
        start = end;
    }
}

fn readLogits(m: *model.Model, output: []u16) !void {
    try m.s.synchronize();
    try (cuda.DeviceBuffer{ .d = m.d, .ptr = m.x.logits, .len = output.len * 2 }).download(0, std.mem.sliceAsBytes(output));
}

fn mixedProbe(a: std.mem.Allocator, io: std.Io, m: *model.Model, ids: []const u32, features: []const f32, frames: usize, audio_at: usize, audio_rows: usize, history: []const u32, dir: []const u8) !void {
    if (history.len == 0) return error.NoTeacherTokens;
    var enc = try Encoder.init(m);
    const audio = try enc.run(m, features, frames);
    for (0..8) |slot| {
        try prepareSlot(m, slot, ids, audio, audio_at, audio_rows);
        for (0..slot) |step| try m.decode(&.{history[step % history.len]}, &.{@intCast(ids.len + step)}, &.{@intCast(slot)});
    }
    const width = m.w.c.vocab;
    const reference = try a.alloc(u16, 8 * width);
    const actual = try a.alloc(u16, 8 * width);
    const MixedRecord = struct { step: usize, batch: usize, permutation: usize, slot: i32, position: i32, different_columns: usize };
    var records: std.ArrayList(MixedRecord) = .empty;
    var failures: usize = 0;
    for (history, 0..) |_, step| {
        var tokens: [8]u32 = undefined;
        var positions: [8]i32 = undefined;
        for (0..8) |slot| {
            tokens[slot] = history[(slot + step) % history.len];
            positions[slot] = @intCast(ids.len + slot + step);
            try m.decode(&.{tokens[slot]}, &.{positions[slot]}, &.{@intCast(slot)});
            try readLogits(m, reference[slot * width ..][0..width]);
        }
        for (0..3) |permutation| {
            var order: [8]usize = undefined;
            for (&order, 0..) |*slot, i| slot.* = switch (permutation) {
                0 => i,
                1 => 7 - i,
                else => (i + 3) % 8,
            };
            for ([_]usize{ 1, 2, 4, 8 }) |batch| {
                var start: usize = 0;
                while (start < 8) : (start += batch) {
                    var batch_ids: [8]u32 = undefined;
                    var batch_pos: [8]i32 = undefined;
                    var batch_slots: [8]i32 = undefined;
                    for (order[start..][0..batch], 0..) |slot, row| {
                        batch_ids[row] = tokens[slot];
                        batch_pos[row] = positions[slot];
                        batch_slots[row] = @intCast(slot);
                    }
                    try m.decode(batch_ids[0..batch], batch_pos[0..batch], batch_slots[0..batch]);
                    try readLogits(m, actual[0 .. batch * width]);
                    for (order[start..][0..batch], 0..) |slot, row| {
                        var different: usize = 0;
                        for (reference[slot * width ..][0..width], actual[row * width ..][0..width]) |old, new| different += @intFromBool(old != new);
                        if (different != 0) failures += 1;
                        try records.append(a, .{ .step = step, .batch = batch, .permutation = permutation, .slot = @intCast(slot), .position = positions[slot], .different_columns = different });
                    }
                }
            }
        }
    }
    for ([_]usize{ 3, 7 }) |slot| try prepareSlot(m, slot, ids, audio, audio_at, audio_rows);
    try m.decode(&.{history[0]}, &.{@intCast(ids.len)}, &.{3});
    try readLogits(m, reference[0..width]);
    try m.decode(&.{ history[0], history[0] }, &.{ @intCast(ids.len), @intCast(ids.len) }, &.{ 7, 3 });
    try readLogits(m, actual[0 .. 2 * width]);
    const reuse_equal = std.mem.eql(u16, reference[0..width], actual[width .. 2 * width]);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const output = try std.json.Stringify.valueAlloc(a, .{ .records = records.items, .failures = failures, .slot_reuse_equal = reuse_equal, .logical_batches = .{ 1, 2, 4, 8 }, .permutations = 3, .context_offsets = .{ 0, 1, 2, 3, 4, 5, 6, 7 } }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "mixed-probe.json" }), .data = output });
    std.debug.print("{s} mixed row invariance: {d}/{d} rows differ; reused slot equal: {}\n", .{ if (failures == 0 and reuse_equal) "PASS" else "FAIL", failures, records.items.len, reuse_equal });
    if (failures != 0 or !reuse_equal) return error.RowArithmeticChanged;
}
