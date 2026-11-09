static bool sd_emu_read(const std::string& path, std::vector<char>& out) {
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

struct SdEmuLayer {
    int din         = 0;
    int dout        = 0;
    float in_scale  = 0.f;
    float out_scale = 0.f;
    std::vector<float> inv;
    std::vector<int8_t> w;
    std::vector<float> ws;
};

struct SdEmuStats {
    long layers = 0;
    long calls  = 0;
    ~SdEmuStats() {
        if (calls > 0) {
            fprintf(stderr, "[emu] layers %ld calls %ld\n", layers, calls);
        }
    }
};
static SdEmuStats g_sd_emu_stats;

static std::unordered_map<std::string, SdEmuLayer*>& sd_emu_reg() {
    static auto* m = new std::unordered_map<std::string, SdEmuLayer*>();
    return *m;
}

static std::vector<int> sd_emu_list(const char* name) {
    std::vector<int> v;
    const char* e = getenv(name);
    if (e == nullptr) {
        return v;
    }
    std::string s(e);
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) {
            j = s.size();
        }
        v.push_back(atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

static SdEmuLayer* sd_emu_load(const std::string& shortn) {
    const char* pack = getenv("SD_EMU_PACK");
    if (pack == nullptr) {
        return nullptr;
    }
    std::string base = std::string(pack) + "/" + shortn;
    std::ifstream m(base + ".meta");
    if (!m) {
        return nullptr;
    }
    SdEmuLayer* L = new SdEmuLayer();
    int T         = 0;
    m >> L->din >> L->dout >> T >> L->in_scale >> L->out_scale;
    std::vector<char> raw, rawws, raws;
    if (!sd_emu_read(base + ".w", raw) || !sd_emu_read(base + ".ws", rawws) || !sd_emu_read(base + ".s", raws)) {
        return nullptr;
    }
    L->w.resize(raw.size());
    memcpy(L->w.data(), raw.data(), raw.size());
    L->ws.resize(rawws.size() / 4);
    memcpy(L->ws.data(), rawws.data(), rawws.size());
    std::vector<float> s(raws.size() / 4);
    memcpy(s.data(), raws.data(), raws.size());
    if ((long)L->w.size() != (long)L->din * L->dout || (int)L->ws.size() != L->dout || (int)s.size() != L->din) {
        fprintf(stderr, "[emu] bad pack sizes for %s\n", shortn.c_str());
        return nullptr;
    }
    L->inv.resize((size_t)L->din);
    for (int j = 0; j < L->din; j++) {
        L->inv[(size_t)j] = 1.0f / (s[(size_t)j] * L->in_scale);
    }
    g_sd_emu_stats.layers++;
    return L;
}

static void sd_emu_par(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    const int nt      = 4;
    const int64_t chk = (n + nt - 1) / nt;
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) {
        int64_t a = t * chk;
        int64_t b = std::min<int64_t>(n, a + chk);
        if (a >= b) {
            break;
        }
        th.emplace_back(fn, a, b);
    }
    for (auto& x : th) {
        x.join();
    }
}

static void sd_emu_op(struct ggml_tensor* dst, int ith, int nth, void* ud) {
    (void)nth;
    if (ith != 0) {
        return;
    }
    SdEmuLayer* L        = (SdEmuLayer*)ud;
    const ggml_tensor* a = dst->src[0];
    const int64_t din    = a->ne[0];
    const int64_t dout   = dst->ne[0];
    const int64_t n1     = a->ne[1];
    const int64_t n2     = a->ne[2];
    const int64_t rows   = n1 * n2 * a->ne[3];
    std::vector<int8_t> q((size_t)rows * (size_t)din);
    for (int64_t r = 0; r < rows; r++) {
        const char* src = (const char*)a->data + (r % n1) * a->nb[1] + ((r / n1) % n2) * a->nb[2] + (r / (n1 * n2)) * a->nb[3];
        int8_t* qr      = &q[(size_t)r * (size_t)din];
        for (int64_t j = 0; j < din; j++) {
            float v = *(const float*)(src + j * a->nb[0]) * L->inv[(size_t)j];
            long iv = lrintf(v);
            qr[j]   = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : iv));
        }
    }
    auto work = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            const int8_t* qr = &q[(size_t)r * (size_t)din];
            char* o          = (char*)dst->data + (r % n1) * dst->nb[1] + ((r / n1) % n2) * dst->nb[2] + (r / (n1 * n2)) * dst->nb[3];
            for (int64_t j = 0; j < dout; j++) {
                const int8_t* wj = &L->w[(size_t)j * (size_t)din];
                int32_t acc      = 0;
                for (int64_t k = 0; k < din; k++) {
                    acc += (int32_t)qr[k] * (int32_t)wj[k];
                }
                float y = (float)acc * L->in_scale * L->ws[(size_t)j];
                long iq = lrintf(y / L->out_scale);
                iq      = iq > 127 ? 127 : (iq < -127 ? -127 : iq);
                *(float*)(o + j * dst->nb[0]) = (float)iq * L->out_scale;
            }
        }
    };
    sd_emu_par(rows, work);
    g_sd_emu_stats.calls++;
}

static bool sd_emu_short(const char* nm, int* block, std::string* shortn) {
    const char* p = strstr(nm, "net.blocks.");
    if (p == nullptr || strstr(nm, "adaln") != nullptr || strstr(nm, "diffusion_model") == nullptr) {
        return false;
    }
    p += strlen("net.blocks.");
    int b           = atoi(p);
    const char* dot = strchr(p, '.');
    if (dot == nullptr) {
        return false;
    }
    std::string rest(dot + 1);
    size_t pos = rest.rfind(".weight");
    if (pos == std::string::npos) {
        return false;
    }
    rest = rest.substr(0, pos);
    for (auto& c : rest) {
        if (c == '.') {
            c = '_';
        }
    }
    *block  = b;
    *shortn = "b" + std::to_string(b) + "_" + rest;
    return true;
}

static ggml_tensor* sd_emu_try(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float scale) {
    static const char* pack = getenv("SD_EMU_PACK");
    if (pack == nullptr || w == nullptr || b != nullptr || scale != 1.f || x->type != GGML_TYPE_F32) {
        return nullptr;
    }
    int block = 0;
    std::string shortn;
    if (!sd_emu_short(w->name, &block, &shortn)) {
        return nullptr;
    }
    static std::vector<int> only = sd_emu_list("SD_EMU_BLOCKS");
    static std::vector<int> skip = sd_emu_list("SD_EMU_SKIP");
    if (!only.empty() && std::find(only.begin(), only.end(), block) == only.end()) {
        return nullptr;
    }
    if (std::find(skip.begin(), skip.end(), block) != skip.end()) {
        return nullptr;
    }
    auto& reg     = sd_emu_reg();
    auto it       = reg.find(shortn);
    SdEmuLayer* L = nullptr;
    if (it == reg.end()) {
        L           = sd_emu_load(shortn);
        reg[shortn] = L;
    } else {
        L = it->second;
    }
    if (L == nullptr || L->din != (int)x->ne[0] || L->dout != (int)w->ne[1]) {
        return nullptr;
    }
    ggml_tensor* args[1] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, L->dout, x->ne[1], x->ne[2], x->ne[3], args, 1, sd_emu_op, 1, L);
}

static ggml_tensor* ggml_ext_linear_orig(ggml_context* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* w,
                                         ggml_tensor* b,
                                         bool force_prec_f32,
                                         float scale);

ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                             ggml_tensor* x,
                             ggml_tensor* w,
                             ggml_tensor* b,
                             bool force_prec_f32,
                             float scale) {
    ggml_tensor* e = sd_emu_try(ctx, x, w, b, scale);
    if (e != nullptr) {
        return e;
    }
    return ggml_ext_linear_orig(ctx, x, w, b, force_prec_f32, scale);
}

// END_EMU
