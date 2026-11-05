#define _POSIX_C_SOURCE 200809L
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { const uint8_t *data; size_t size, at; } Memory;

static int read_memory(void *opaque, uint8_t *dst, int count) {
    Memory *m = opaque;
    if (m->at == m->size) return AVERROR_EOF;
    size_t n = m->size - m->at;
    if (n > (size_t)count) n = count;
    memcpy(dst, m->data + m->at, n);
    m->at += n;
    return (int)n;
}

static int64_t seek_memory(void *opaque, int64_t offset, int whence) {
    Memory *m = opaque;
    if (whence == AVSEEK_SIZE) return (int64_t)m->size;
    whence &= ~AVSEEK_FORCE;
    int64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? (int64_t)m->at :
                   whence == SEEK_END ? (int64_t)m->size : -1;
    if (base < 0 || offset < -base || offset > (int64_t)m->size - base) return -1;
    m->at = (size_t)(base + offset);
    return (int64_t)m->at;
}

void tf_audio_free(float *data) { av_free(data); }
const char *tf_audio_version(void) { return av_version_info(); }

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int decode_audio(const uint8_t *data, size_t size, float **out, size_t *length, double *timing) {
    const double begin = timing ? seconds() : 0;
    double resample = 0;
    const int capacity = 600 * 16000;
    *out = NULL;
    *length = 0;
    if (!data || !size || size > 25 * 1024 * 1024) return -1;
    Memory memory = {data, size, 0};
    AVFormatContext *fmt = avformat_alloc_context();
    AVIOContext *io = NULL;
    AVCodecContext *codec = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    SwrContext *swr = NULL;
    float *samples = NULL;
    int result = -1, used = 0;
    if (!fmt) goto cleanup;
    uint8_t *buffer = av_malloc(65536);
    if (!buffer) goto cleanup;
    io = avio_alloc_context(buffer, 65536, 0, &memory, read_memory, NULL, seek_memory);
    if (!io) { av_free(buffer); goto cleanup; }
    fmt->pb = io;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
    AVDictionary *options = NULL;
    av_dict_set(&options, "format_whitelist", "wav,flac,mp3,mov,ogg,matroska,aac", 0);
    av_dict_set(&options, "protocol_whitelist", "pipe", 0);
    int opened = avformat_open_input(&fmt, NULL, NULL, &options);
    av_dict_free(&options);
    if (opened < 0) goto cleanup;
    if (avformat_find_stream_info(fmt, NULL) < 0) goto cleanup;
    const AVCodec *decoder = NULL;
    int stream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
    if (stream < 0 || !decoder) goto cleanup;
    codec = avcodec_alloc_context3(decoder);
    if (!codec || avcodec_parameters_to_context(codec, fmt->streams[stream]->codecpar) < 0) goto cleanup;
    codec->thread_count = 1;
    if (avcodec_open2(codec, decoder, NULL) < 0 || codec->sample_rate <= 0) goto cleanup;
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_FLT, 16000, &codec->ch_layout,
                           codec->sample_fmt, codec->sample_rate, 0, NULL) < 0) goto cleanup;
    int channels = codec->ch_layout.nb_channels;
    if (channels < 1 || channels > 128) goto cleanup;
    double matrix[128];
    for (int i = 0; i < channels; ++i) matrix[i] = 1.0 / channels;
    if (swr_set_matrix(swr, matrix, channels) < 0) goto cleanup;
    if (swr_init(swr) < 0) goto cleanup;
    samples = av_malloc_array((size_t)capacity + 1, sizeof(float));
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!samples || !packet || !frame) goto cleanup;
    int eof = 0;
    while (!eof) {
        int rc = av_read_frame(fmt, packet);
        if (rc < 0) {
            if (rc != AVERROR_EOF) goto cleanup;
            eof = 1;
            rc = avcodec_send_packet(codec, NULL);
        } else if (packet->stream_index == stream) rc = avcodec_send_packet(codec, packet);
        else { av_packet_unref(packet); continue; }
        av_packet_unref(packet);
        if (rc < 0 && rc != AVERROR_EOF) goto cleanup;
        while ((rc = avcodec_receive_frame(codec, frame)) >= 0) {
            uint8_t *dst = (uint8_t *)(samples + used);
            double before = timing ? seconds() : 0;
            int n = swr_convert(swr, &dst, capacity + 1 - used, (const uint8_t **)frame->extended_data, frame->nb_samples);
            if (timing) resample += seconds() - before;
            av_frame_unref(frame);
            if (n < 0 || used + n > capacity) { result = -2; goto cleanup; }
            used += n;
        }
        if (rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) goto cleanup;
    }
    for (;;) {
        uint8_t *dst = (uint8_t *)(samples + used);
        double before = timing ? seconds() : 0;
        int n = swr_convert(swr, &dst, capacity + 1 - used, NULL, 0);
        if (timing) resample += seconds() - before;
        if (n < 0 || used + n > capacity) { result = -2; goto cleanup; }
        used += n;
        if (!n) break;
    }
    if (used <= 200) goto cleanup;
    for (int i = 0; i < used; ++i) if (!isfinite(samples[i])) goto cleanup;
    *out = samples;
    *length = (size_t)used;
    samples = NULL;
    result = 0;
cleanup:
    av_free(samples);
    swr_free(&swr);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    avformat_close_input(&fmt);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    if (timing) { timing[0] = seconds() - begin - resample; timing[1] = resample; }
    return result;
}

int tf_audio_decode(const uint8_t *data, size_t size, float **out, size_t *length) {
    return decode_audio(data, size, out, length, NULL);
}

int tf_audio_decode_timed(const uint8_t *data, size_t size, float **out, size_t *length, double timing[2]) {
    return decode_audio(data, size, out, length, timing);
}
