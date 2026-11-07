static double sd_now_ms() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

static bool sd_read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(n);
    f.read(out.data(), (std::streamsize)n);
    return (bool)f;
}

static const char* sd_mmap_file(const std::string& path, size_t* size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }
    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    *size = (size_t)st.st_size;
    return (const char*)p;
}

static void sd_log_mem(const char* tag) {
    long rss = 0, avail = 0;
    std::string line;
    std::ifstream f("/proc/self/status");
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            rss = atol(line.c_str() + 6);
            break;
        }
    }
    std::ifstream g("/proc/meminfo");
    while (std::getline(g, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            avail = atol(line.c_str() + 13);
            break;
        }
    }
    fprintf(stderr, "[npu] mem %s rss_mb %ld avail_mb %ld\n", tag, rss / 1024, avail / 1024);
}

static void sd_mix_bytes(uint64_t h[4], const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    size_t i         = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, b + i, 8);
        for (int k = 0; k < 4; k++) {
            h[k] ^= v + 0x9E3779B97F4A7C15ULL * (uint64_t)(k + 1) + (h[k] << 6) + (h[k] >> 2);
        }
    }
    for (; i < n; i++) {
        for (int k = 0; k < 4; k++) {
            h[k] ^= b[i];
            h[k] *= 1099511628211ULL;
        }
    }
}

static void sd_quant_row(const float* x, const float* inv, int8_t* q, int64_t n) {
    int64_t j          = 0;
    const int32x4_t lo = vdupq_n_s32(-127);
    const int32x4_t hi = vdupq_n_s32(127);
    for (; j + 8 <= n; j += 8) {
        float32x4_t a0 = vmulq_f32(vld1q_f32(x + j), vld1q_f32(inv + j));
        float32x4_t a1 = vmulq_f32(vld1q_f32(x + j + 4), vld1q_f32(inv + j + 4));
        int32x4_t i0   = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a0), lo), hi);
        int32x4_t i1   = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a1), lo), hi);
        int16x8_t s16  = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        vst1_s8(q + j, vqmovn_s16(s16));
    }
    for (; j < n; j++) {
        long iv = lrintf(x[j] * inv[j]);
        q[j]    = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : iv));
    }
}

static void sd_dequant_row(const int8_t* q, float scale, float* o, int64_t n) {
    int64_t j            = 0;
    const float32x4_t sc = vdupq_n_f32(scale);
    for (; j + 8 <= n; j += 8) {
        int16x8_t s16  = vmovl_s8(vld1_s8(q + j));
        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16))), sc);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16))), sc);
        vst1q_f32(o + j, f0);
        vst1q_f32(o + j + 4, f1);
    }
    for (; j < n; j++) {
        o[j] = (float)q[j] * scale;
    }
}

struct SdNpuLayer {
    int din         = 0;
    int dout        = 0;
    int T           = 256;
    float in_scale  = 0.f;
    float out_scale = 0.f;
    std::vector<float> inv;
    const char* w = nullptr;
    size_t wsize  = 0;
    std::vector<float> ws;
    std::vector<int32_t> bias;
    std::vector<int8_t> qin;
    std::vector<int8_t> qout;
    ANeuralNetworksModel* model      = nullptr;
    ANeuralNetworksCompilation* comp = nullptr;
};

struct SdNpuStats {
    double npu_ms  = 0;
    double prep_ms = 0;
    long calls     = 0;
    long tiles     = 0;
    int loaded     = 0;
    int failed     = 0;
    ~SdNpuStats() {
        if (calls > 0 || loaded > 0) {
            fprintf(stderr, "[npu] layers loaded %d failed %d\n", loaded, failed);
            fprintf(stderr, "[npu] calls %ld tiles %ld npu_ms %.0f prep_ms %.0f\n", calls, tiles, npu_ms, prep_ms);
        }
    }
};
static SdNpuStats g_sd_npu_stats;

static std::unordered_map<std::string, SdNpuLayer*>& sd_npu_reg() {
    static auto* m = new std::unordered_map<std::string, SdNpuLayer*>();
    return *m;
}

static const ANeuralNetworksDevice* sd_npu_device() {
    static const ANeuralNetworksDevice* dev = nullptr;
    static bool done                        = false;
    if (!done) {
        done             = true;
        const char* want = getenv("SD_NPU_DEV") ? getenv("SD_NPU_DEV") : "mtk-mdla";
        uint32_t nd      = 0;
        ANeuralNetworks_getDeviceCount(&nd);
        for (uint32_t i = 0; i < nd; i++) {
            ANeuralNetworksDevice* d = nullptr;
            ANeuralNetworks_getDevice(i, &d);
            const char* nm = nullptr;
            ANeuralNetworksDevice_getName(d, &nm);
            if (nm != nullptr && strcmp(nm, want) == 0) {
                dev = d;
            }
        }
        fprintf(stderr, "[npu] device %s %s\n", want, dev ? "found" : "NOT FOUND");
    }
    return dev;
}

#define SD_NCK(x)                                                                       \
    do {                                                                                \
        int r_ = (x);                                                                   \
        if (r_ != ANEURALNETWORKS_NO_ERROR) {                                           \
            fprintf(stderr, "[npu] NNAPI error %d at line %d: %s\n", r_, __LINE__, #x); \
            g_sd_npu_stats.failed++;                                                    \
            return nullptr;                                                             \
        }                                                                               \
    } while (0)

// END_NPU_A
