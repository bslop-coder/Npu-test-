static SdNpuLayer* sd_npu_load(const std::string& shortn) {
    const char* pack  = getenv("SD_NPU_PACK");
    const char* cache = getenv("SD_NPU_CACHE");
    if (pack == nullptr || cache == nullptr) {
        return nullptr;
    }
    const ANeuralNetworksDevice* dev = sd_npu_device();
    if (dev == nullptr) {
        return nullptr;
    }
    std::string base = std::string(pack) + "/" + shortn;
    std::ifstream m(base + ".meta");
    if (!m) {
        return nullptr;
    }
    SdNpuLayer* L = new SdNpuLayer();
    m >> L->din >> L->dout >> L->T >> L->in_scale >> L->out_scale;
    std::vector<char> raw;
    std::vector<float> s;
    L->w = sd_mmap_file(base + ".w", &L->wsize);
    if (L->w == nullptr || !sd_read_file(base + ".ws", raw)) {
        return nullptr;
    }
    L->ws.resize(raw.size() / 4);
    memcpy(L->ws.data(), raw.data(), raw.size());
    if (!sd_read_file(base + ".s", raw)) {
        return nullptr;
    }
    s.resize(raw.size() / 4);
    memcpy(s.data(), raw.data(), raw.size());
    if ((long)L->wsize != (long)L->din * L->dout || (int)L->ws.size() != L->dout || (int)s.size() != L->din) {
        fprintf(stderr, "[npu] bad pack sizes for %s\n", shortn.c_str());
        return nullptr;
    }
    L->inv.resize((size_t)L->din);
    for (int j = 0; j < L->din; j++) {
        L->inv[(size_t)j] = 1.0f / (s[(size_t)j] * L->in_scale);
    }
    L->qin.assign((size_t)L->T * L->din, 0);
    L->qout.assign((size_t)L->T * L->dout, 0);
    L->bias.assign((size_t)L->dout, 0);
    SD_NCK(ANeuralNetworksModel_create(&L->model));
    uint32_t in_dims[4]  = {1, (uint32_t)L->T, 1, (uint32_t)L->din};
    uint32_t w_dims[4]   = {(uint32_t)L->dout, 1, 1, (uint32_t)L->din};
    uint32_t b_dims[1]   = {(uint32_t)L->dout};
    uint32_t out_dims[4] = {1, (uint32_t)L->T, 1, (uint32_t)L->dout};
    ANeuralNetworksOperandType t_in  = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, in_dims, L->in_scale, 0};
    ANeuralNetworksOperandType t_w   = {ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL, 4, w_dims, 0.0f, 0};
    ANeuralNetworksOperandType t_b   = {ANEURALNETWORKS_TENSOR_INT32, 1, b_dims, 0.0f, 0};
    ANeuralNetworksOperandType t_i32 = {ANEURALNETWORKS_INT32, 0, nullptr, 0.0f, 0};
    ANeuralNetworksOperandType t_out = {ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, 4, out_dims, L->out_scale, 0};
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_in));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_w));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_b));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_i32));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_i32));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_i32));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_i32));
    SD_NCK(ANeuralNetworksModel_addOperand(L->model, &t_out));
    ANeuralNetworksSymmPerChannelQuantParams qp;
    qp.channelDim = 0;
    qp.scaleCount = (uint32_t)L->dout;
    qp.scales     = L->ws.data();
    SD_NCK(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(L->model, 1, &qp));
    int32_t pad = ANEURALNETWORKS_PADDING_VALID;
    int32_t one = 1;
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 1, L->w, L->wsize));
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 2, L->bias.data(), L->bias.size() * 4));
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 3, &pad, 4));
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 4, &one, 4));
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 5, &one, 4));
    SD_NCK(ANeuralNetworksModel_setOperandValue(L->model, 6, &act, 4));
    uint32_t op_in[7]  = {0, 1, 2, 3, 4, 5, 6};
    uint32_t op_out[1] = {7};
    SD_NCK(ANeuralNetworksModel_addOperation(L->model, ANEURALNETWORKS_CONV_2D, 7, op_in, 1, op_out));
    uint32_t m_in[1]  = {0};
    uint32_t m_out[1] = {7};
    SD_NCK(ANeuralNetworksModel_identifyInputsAndOutputs(L->model, 1, m_in, 1, m_out));
    SD_NCK(ANeuralNetworksModel_finish(L->model));
    const ANeuralNetworksDevice* devs[1] = {dev};
    SD_NCK(ANeuralNetworksCompilation_createForDevices(L->model, devs, 1, &L->comp));
    SD_NCK(ANeuralNetworksCompilation_setPreference(L->comp, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED));
    uint64_t h[4] = {1469598103934665603ULL, 0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL};
    sd_mix_bytes(h, shortn.data(), shortn.size());
    sd_mix_bytes(h, &L->in_scale, 4);
    sd_mix_bytes(h, &L->out_scale, 4);
    sd_mix_bytes(h, &L->T, 4);
    sd_mix_bytes(h, L->ws.data(), L->ws.size() * 4);
    sd_mix_bytes(h, L->w, L->wsize);
    uint8_t token[32];
    memcpy(token, h, 32);
    SD_NCK(ANeuralNetworksCompilation_setCaching(L->comp, cache, token));
    SD_NCK(ANeuralNetworksCompilation_finish(L->comp));
    g_sd_npu_stats.loaded++;
    if (shortn.size() > 10 && shortn.compare(shortn.size() - 10, 10, "mlp_layer2") == 0) {
        sd_log_mem(shortn.c_str());
    }
    return L;
}

// END_NPU_B
