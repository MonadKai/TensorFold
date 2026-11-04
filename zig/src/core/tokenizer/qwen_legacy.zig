const std = @import("std");

pub fn read(io: std.Io, a: std.mem.Allocator, dir: []const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp = arena.allocator();
    const config = try file(io, tmp, dir, "tokenizer_config.json");
    const c = try std.json.parseFromSliceLeaky(std.json.Value, tmp, config, .{});
    if (c != .object) return error.UnsupportedTokenizer;
    const cls = c.object.get("tokenizer_class") orelse return error.UnsupportedTokenizer;
    if (cls != .string or !std.mem.eql(u8, cls.string, "Qwen2Tokenizer")) return error.UnsupportedTokenizer;
    const vocab = try file(io, tmp, dir, "vocab.json");
    const merges = try file(io, tmp, dir, "merges.txt");
    const raw = try std.fmt.allocPrint(tmp, "{{\"model\":{{\"type\":\"BPE\",\"vocab\":{s},\"merges\":[]}},\"added_tokens\":[]," ++
        "\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\\\p{{L}}\\\\p{{N}}]?\\\\p{{L}}+|\\\\p{{N}}| ?[^\\\\s\\\\p{{L}}\\\\p{{N}}]+[\\r\\n]*|\\\\s*[\\r\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"}},\"behavior\":\"Isolated\",\"invert\":false}},{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}}]}}," ++
        "\"decoder\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}}}}", .{vocab});
    var root = try std.json.parseFromSliceLeaky(std.json.Value, tmp, raw, .{});
    var list: std.array_list.Managed(std.json.Value) = .init(tmp);
    var lines = std.mem.splitScalar(u8, merges, '\n');
    while (lines.next()) |line| {
        const clean = std.mem.trim(u8, line, "\r");
        if (clean.len == 0 or clean[0] == '#') continue;
        try list.append(.{ .string = clean });
    }
    root.object.getPtr("model").?.object.getPtr("merges").?.* = .{ .array = list };
    var added: std.array_list.Managed(std.json.Value) = .init(tmp);
    const table = c.object.get("added_tokens_decoder") orelse return error.UnsupportedTokenizer;
    if (table != .object) return error.UnsupportedTokenizer;
    var it = table.object.iterator();
    while (it.next()) |entry| {
        var token = entry.value_ptr.*;
        if (token != .object) return error.UnsupportedTokenizer;
        const id = try std.fmt.parseInt(u32, entry.key_ptr.*, 10);
        try token.object.put(tmp, "id", .{ .integer = id });
        try added.append(token);
    }
    root.object.getPtr("added_tokens").?.* = .{ .array = added };
    return std.json.Stringify.valueAlloc(a, root, .{});
}

fn file(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
}
