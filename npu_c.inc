static void sd_npu_op(struct ggml_tensor* dst, int ith, int nth, void* ud) {
    (void)nth;
    if (ith != 0) {
        return;
    }
    SdNpuLayer* L        = (SdNpuLayer*)ud;
    const ggml_tensor* a = dst->src[0];
    const int64_t din    = a->ne[0];
    const int64_t dout   = dst->ne[0];
    const int64_t n1     = a->ne[1];
    const int64_t n2     = a->ne[2];
    const int64_t rows   = n1 * n2 * a->ne[3];
    const bool fast_in   = a->nb[0] == sizeof(float);
    const bool fast_out  = dst->nb[0] == sizeof(float);
    for (int64_t r0 = 0; r0 < rows; r0 += L->T) {
        double t0        = sd_now_ms();
        const int64_t nr = std::min<int64_t>(L->T, rows - r0);
        for (int64_t r = 0; r < L->T; r++) {
            int8_t* q = &L->qin[(size_t)r * (size_t)din];
            if (r >= nr) {
                memset(q, 0, (size_t)din);
                continue;
            }
            const int64_t rr = r0 + r;
            const char* src  = (const char*)a->data + (rr % n1) * a->nb[1] + ((rr / n1) % n2) * a->nb[2] + (rr / (n1 * n2)) * a->nb[3];
            if (fast_in) {
                sd_quant_row((const float*)src, L->inv.data(), q, din);
            } else {
                for (int64_t j = 0; j < din; j++) {
                    float v = *(const float*)(src + j * a->nb[0]) * L->inv[(size_t)j];
                    long iv = lrintf(v);
                    q[j]    = (int8_t)(iv > 127 ? 127 : (iv < -127 ? -127 : iv));
                }
            }
        }
        double t1                    = sd_now_ms();
        ANeuralNetworksExecution* ex = nullptr;
        bool ok                      = ANeuralNetworksExecution_create(L->comp, &ex) == ANEURALNETWORKS_NO_ERROR;
        if (ok) {
            ok = ANeuralNetworksExecution_setInput(ex, 0, nullptr, L->qin.data(), L->qin.size()) == ANEURALNETWORKS_NO_ERROR;
        }
        if (ok) {
            ok = ANeuralNetworksExecution_setOutput(ex, 0, nullptr, L->qout.data(), L->qout.size()) == ANEURALNETWORKS_NO_ERROR;
        }
        if (ok) {
            ok = ANeuralNetworksExecution_compute(ex) == ANEURALNETWORKS_NO_ERROR;
        }
        if (ex != nullptr) {
            ANeuralNetworksExecution_free(ex);
        }
        if (!ok) {
            fprintf(stderr, "[npu] execution failed\n");
            abort();
        }
        double t2 = sd_now_ms();
        for (int64_t r = 0; r < nr; r++) {
            const int64_t rr = r0 + r;
            char* o          = (char*)dst->data + (rr % n1) * dst->nb[1] + ((rr / n1) % n2) * dst->nb[2] + (rr / (n1 * n2)) * dst->nb[3];
            const int8_t* q  = &L->qout[(size_t)r * (size_t)dout];
            if (fast_out) {
                sd_dequant_row(q, L->out_scale, (float*)o, dout);
            } else {
                for (int64_t j = 0; j < dout; j++) {
                    *(float*)(o + j * dst->nb[0]) = (float)q[j] * L->out_scale;
                }
            }
        }
        double t3 = sd_now_ms();
        g_sd_npu_stats.prep_ms += (t1 - t0) + (t3 - t2);
        g_sd_npu_stats.npu_ms += (t2 - t1);
        g_sd_npu_stats.tiles++;
    }
    g_sd_npu_stats.calls++;
}

static bool sd_npu_short(const char* nm, int* block, std::string* shortn) {
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

static ggml_tensor* sd_npu_try(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float scale) {
    static const char* pack = getenv("SD_NPU_PACK");
    if (pack == nullptr || w == nullptr || b != nullptr || scale != 1.f || x->type != GGML_TYPE_F32) {
        return nullptr;
    }
    int block = 0;
    std::string shortn;
    if (!sd_npu_short(w->name, &block, &shortn)) {
        return nullptr;
    }
    static std::vector<int> only = []() {
        std::vector<int> v;
        const char* e = getenv("SD_NPU_BLOCKS");
        if (e != nullptr) {
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
        }
        return v;
    }();
    if (!only.empty() && std::find(only.begin(), only.end(), block) == only.end()) {
        return nullptr;
    }
    auto& reg     = sd_npu_reg();
    auto it       = reg.find(shortn);
    SdNpuLayer* L = nullptr;
    if (it == reg.end()) {
        L           = sd_npu_load(shortn);
        reg[shortn] = L;
    } else {
        L = it->second;
    }
    if (L == nullptr || L->din != (int)x->ne[0] || L->dout != (int)w->ne[1]) {
        return nullptr;
    }
    ggml_tensor* args[1] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, L->dout, x->ne[1], x->ne[2], x->ne[3], args, 1, sd_npu_op, 1, L);
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
    ggml_tensor* npu = sd_npu_try(ctx, x, w, b, scale);
    if (npu != nullptr) {
        return npu;
    }
    return ggml_ext_linear_orig(ctx, x, w, b, force_prec_f32, scale);
}

// END_NPU_C
