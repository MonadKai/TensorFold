const std = @import("std");
const sys = @import("sys");

const Fns = struct {
    decode: *const fn ([*]const u8, usize, *?[*]f32, *usize) callconv(.c) c_int,
    free: *const fn ([*]f32) callconv(.c) void,
    version: *const fn () callconv(.c) [*:0]const u8,
};

var mutex: std.atomic.Mutex = .unlocked;
var functions: ?Fns = null;

fn load() !Fns {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
    defer mutex.unlock();
    if (functions) |f| return f;
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    var dl: ?sys.DynLib = null;
    if (sys.optional_libraries.video_library.executableDir(&buffer)) |dir| {
        var path: [std.fs.max_path_bytes]u8 = undefined;
        const p = try std.fmt.bufPrint(&path, "{s}/../lib/libtfaudio.so", .{dir});
        dl = sys.DynLib.open(p) catch null;
    }
    var lib = dl orelse (sys.DynLib.open("libtfaudio.so") catch return error.AudioLibraryUnavailable);
    errdefer lib.close();
    const f: Fns = .{
        .decode = lib.lookup(@TypeOf(@as(Fns, undefined).decode), "tf_audio_decode") orelse return error.AudioLibraryUnavailable,
        .free = lib.lookup(@TypeOf(@as(Fns, undefined).free), "tf_audio_free") orelse return error.AudioLibraryUnavailable,
        .version = lib.lookup(@TypeOf(@as(Fns, undefined).version), "tf_audio_version") orelse return error.AudioLibraryUnavailable,
    };
    functions = f;
    return f;
}

pub fn decode(a: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len == 0 or bytes.len > 25 * 1024 * 1024) return error.AudioUploadLimit;
    const f = try load();
    var ptr: ?[*]f32 = null;
    var n: usize = 0;
    const rc = f.decode(bytes.ptr, bytes.len, &ptr, &n);
    if (rc != 0) return if (rc == -2) error.AudioDurationLimit else error.InvalidAudio;
    defer f.free(ptr.?);
    return a.dupe(f32, ptr.?[0..n]);
}

pub fn available() !void {
    _ = try load();
}
