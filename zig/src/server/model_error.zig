const std = @import("std");
const json = @import("json");

pub fn body(a: std.mem.Allocator, message: []const u8) std.mem.Allocator.Error!json.Value {
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message });
    try o.put(a, "type", .{ .string = "NotFoundError" });
    try o.put(a, "param", .{ .string = "model" });
    try o.put(a, "code", try json.intValue(a, 404));
    return .{ .object = o };
}
