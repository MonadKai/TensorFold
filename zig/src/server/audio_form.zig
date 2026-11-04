const std = @import("std");
const errors = @import("errors.zig");
const Cx = errors.Cx;

pub const Form = struct { fields: std.StringHashMapUnmanaged([]const u8) = .empty, file: []const u8 = "" };

pub fn parse(cx: *Cx, content_type: []const u8, body: []const u8) errors.Refused!Form {
    var params = std.mem.splitScalar(u8, content_type, ';');
    if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, params.next().?, " \t"), "multipart/form-data"))
        return cx.refuse("expected multipart/form-data");
    var boundary: ?[]const u8 = null;
    while (params.next()) |p| {
        const part = std.mem.trim(u8, p, " \t");
        if (std.mem.startsWith(u8, part, "boundary=")) {
            if (boundary != null) return cx.refuse("duplicate multipart boundary");
            boundary = std.mem.trim(u8, part[9..], "\"");
        }
    }
    const b = boundary orelse return cx.refuse("missing multipart boundary");
    if (b.len == 0 or b.len > 70) return cx.refuse("invalid multipart boundary");
    for (b) |c| if (c < 33 or c > 126 or c == '"') return cx.refuse("invalid multipart boundary");
    const delimiter = try std.fmt.allocPrint(cx.a, "--{s}", .{b});
    const separator = try std.fmt.allocPrint(cx.a, "\r\n--{s}", .{b});
    if (!std.mem.startsWith(u8, body, delimiter)) return cx.refuse("malformed multipart body");
    var at = delimiter.len;
    var form: Form = .{};
    var seen_file = false;
    var parts: usize = 0;
    while (true) {
        if (std.mem.startsWith(u8, body[at..], "--")) break;
        if (!std.mem.startsWith(u8, body[at..], "\r\n")) return cx.refuse("malformed multipart boundary");
        at += 2;
        const header_end = std.mem.indexOfPos(u8, body, at, "\r\n\r\n") orelse return cx.refuse("missing multipart headers");
        if (header_end - at > 8192) return cx.refuse("multipart headers exceed limit");
        const header = body[at..header_end];
        var disposition: ?[]const u8 = null;
        var lines = std.mem.splitSequence(u8, header, "\r\n");
        while (lines.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse return cx.refuse("malformed multipart header");
            if (std.ascii.eqlIgnoreCase(line[0..colon], "content-disposition")) {
                if (disposition != null) return cx.refuse("duplicate content-disposition");
                disposition = std.mem.trim(u8, line[colon + 1 ..], " \t");
            }
        }
        var attrs = std.mem.splitScalar(u8, disposition orelse return cx.refuse("missing content-disposition"), ';');
        if (!std.ascii.eqlIgnoreCase(attrs.next().?, "form-data")) return cx.refuse("expected form-data disposition");
        var name: ?[]const u8 = null;
        var filename = false;
        while (attrs.next()) |attr| {
            const text = std.mem.trim(u8, attr, " \t");
            if (std.mem.startsWith(u8, text, "name=\"")) {
                if (name != null or text.len < 7 or text[text.len - 1] != '"') return cx.refuse("invalid multipart field name");
                name = text[6 .. text.len - 1];
            } else if (std.mem.startsWith(u8, text, "filename=")) filename = true;
        }
        const key = name orelse return cx.refuse("missing multipart field name");
        const start = header_end + 4;
        var end = start;
        while (true) {
            end = std.mem.indexOfPos(u8, body, end, separator) orelse return cx.refuse("unterminated multipart field");
            const suffix = body[end + separator.len ..];
            if (std.mem.startsWith(u8, suffix, "--") or std.mem.startsWith(u8, suffix, "\r\n")) break;
            end += separator.len;
        }
        const value = body[start..end];
        if (std.mem.eql(u8, key, "file")) {
            if (seen_file or !filename) return cx.refuse("file must be one uploaded file");
            seen_file = true;
            form.file = value;
            if (value.len > 25 * 1024 * 1024) return cx.refuse("audio upload exceeds 25 MiB");
        } else {
            if (value.len > 16384 or form.fields.contains(key)) return cx.refuse("duplicate or oversized multipart field");
            try form.fields.put(cx.a, key, value);
        }
        parts += 1;
        if (parts > 32) return cx.refuse("too many multipart fields");
        at = end + separator.len;
    }
    if (!seen_file) return cx.refuse("file is required");
    return form;
}

test "multipart keeps binary file bytes and rejects duplicate fields and incomplete bodies" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const body = "--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n\r\na\x00b\r\n--x\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n--x--\r\n";
    const got = try parse(&cx, "multipart/form-data; boundary=\"x\"", body);
    try std.testing.expectEqualStrings("a\x00b", got.file);
    try std.testing.expectEqualStrings("en", got.fields.get("language").?);
    try std.testing.expectError(error.Refused, parse(&cx, "multipart/form-data; boundary=x", body[0 .. body.len - 8]));
}
