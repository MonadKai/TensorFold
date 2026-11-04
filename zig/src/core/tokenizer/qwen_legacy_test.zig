const std = @import("std");
const tokenizer = @import("tokenizer.zig");

test "Qwen2 original assets keep hash-prefixed BPE merges through the actual loader" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"tokenizer_class\":\"Qwen2Tokenizer\",\"added_tokens_decoder\":{}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "vocab.json", .data = "{\"#\":0,\"##\":1,\"####\":2,\"Ã\":3,\"©\":4,\"Ã©\":5,\"e\":6,\"Ì\":7,\"ģ\":8,\"Ìģ\":9}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "merges.txt", .data = "#version: 0.2\n# #\n## ##\nÃ ©\nÌ ģ\n" });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(path);
    var t = try tokenizer.loadTokenizer(io, a, path);
    defer t.deinit();
    for ([_][]const u8{ "##", "####", "é", "e\u{0301}" }, [_]u32{ 1, 2, 5, 5 }) |source, expected| {
        const ids = try t.encode(a, source);
        defer a.free(ids);
        try std.testing.expectEqualSlices(u32, &.{expected}, ids);
    }
}
