//! Optional audio-family GPU event intervals; a shared batch is charged whole to every participating request.
pub const Sum = struct {
    encoder_seconds: f64 = 0,
    prefill_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    head_seconds: f64 = 0,
    sample_seconds: f64 = 0,
    decode_calls: u64 = 0,
    decode_rows: u64 = 0,

    pub fn add(sum: *Sum, other: Sum) void {
        inline for (@typeInfo(Sum).@"struct".field_names) |name| @field(sum, name) += @field(other, name);
    }
};
