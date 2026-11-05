//! Child telemetry must outlive the child's serialized buffer and remain owned by the parent request.
const std = @import("std");
const json = @import("json");

pub fn copy(a: std.mem.Allocator, bytes: []const u8) std.mem.Allocator.Error!json.Value {
    const owned = try a.dupe(u8, bytes);
    return (try json.parse(a, owned)).ok;
}

test "audio telemetry owns nested strings and integer text after the child buffer is reclaimed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const buffer = try std.testing.allocator.dupe(u8, "{\"engine\":\"lanes\",\"nested\":{\"token_ids\":[11528]}}");
    defer std.testing.allocator.free(buffer);
    const value = try copy(arena.allocator(), buffer);
    @memset(buffer, 'x');
    try std.testing.expectEqualStrings("lanes", value.strField("engine") orelse "");
    try std.testing.expectEqual(@as(?i64, 11528), value.get("nested").?.get("token_ids").?.array[0].int64());
}
