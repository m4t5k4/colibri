struct KdaProto {
    int device, H, D, P, hidden, K, committed;
    cudaStream_t stream;
    ColiCudaTensor *proj[4];
    float *state[2], *window[2];
    float *conv, *norm, *input, *host_input;
    float *x, *decay, *beta, *gate;
    float *qkv, *mixed, *core, *normed, *out;
    size_t projection_bytes, state_bytes, transactional_state_bytes;
    size_t window_bytes, scratch_bytes, other_bytes, host_staging_bytes;
    int profile;
    cudaEvent_t mark[7];
    ColiCudaKdaTimes times;
};

static bool kda_alloc(float **p, size_t bytes, size_t *account) {
    *p = nullptr;
    if (cudaMalloc((void **)p, bytes) != cudaSuccess) return false;
    *account += coli_cuda_alloc_footprint(bytes);
    return true;
}

static void kda_free(KdaProto *a) {
    if (!a) return;
    cudaSetDevice(a->device);
    if (a->stream) cudaStreamSynchronize(a->stream);
    for (int i = 0; i < 4; i++) coli_cuda_tensor_free(a->proj[i]);
    for (int i = 0; i < 2; i++) { cudaFree(a->state[i]); cudaFree(a->window[i]); }
    cudaFree(a->conv); cudaFree(a->norm); cudaFree(a->input);
    if (a->host_input) cudaFreeHost(a->host_input);
    cudaFree(a->qkv); cudaFree(a->mixed); cudaFree(a->core);
    cudaFree(a->normed); cudaFree(a->out);
    for (int i = 0; i < 7; i++) if (a->mark[i]) cudaEventDestroy(a->mark[i]);
    if (a->stream) cudaStreamDestroy(a->stream);
    delete a;
}

static KdaProto *kda_create(int device, int H, int D, int hidden, int K,
                            const ColiCudaKdaMatrix *mat, const float *conv,
                            const float *norm) {
    if (H < 2 || D < 2 || K != 4 || hidden % 64 || (H * D) % 64) return nullptr;
    KdaProto *a = new KdaProto{};
    a->device = device; a->H = H; a->D = D; a->P = H * D;
    a->hidden = hidden; a->K = K;
    size_t input_bytes = ((size_t)hidden + 2u * a->P + H) * sizeof(float);
    if (cudaSetDevice(device) != cudaSuccess) goto fail;
    for (int i = 0; i < 4; i++) {
        if (mat[i].in != (i == 3 ? a->P : hidden) ||
            mat[i].out != (i == 3 ? hidden : a->P) ||
            !coli_cuda_tensor_upload_g(&a->proj[i], mat[i].q4,
                                       mat[i].scales, 4, mat[i].in,
                                       mat[i].out, device, 64)) goto fail;
        a->projection_bytes += coli_cuda_tensor_vram(a->proj[i]);
    }
    /* Tensor upload converts nibbles on stream 0; establish the dependency
     * once, before the object's independent nonblocking stream begins. */
    if (cudaStreamSynchronize(0) != cudaSuccess ||
        cudaStreamCreateWithFlags(&a->stream, cudaStreamNonBlocking) != cudaSuccess)
        goto fail;
    if (!kda_alloc(&a->state[0], (size_t)H * D * D * 4, &a->state_bytes) ||
        !kda_alloc(&a->state[1], (size_t)H * D * D * 4, &a->transactional_state_bytes) ||
        !kda_alloc(&a->window[0], (size_t)3 * a->P * K * 4, &a->window_bytes) ||
        !kda_alloc(&a->window[1], (size_t)3 * a->P * K * 4,
                   &a->transactional_state_bytes) ||
        !kda_alloc(&a->conv, (size_t)3 * a->P * K * 4, &a->other_bytes) ||
        !kda_alloc(&a->norm, (size_t)D * 4, &a->other_bytes) ||
        !kda_alloc(&a->input, input_bytes, &a->scratch_bytes) ||
        !kda_alloc(&a->qkv, (size_t)3 * a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->mixed, (size_t)3 * a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->core, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->normed, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->out, (size_t)hidden * 4, &a->scratch_bytes)) goto fail;
    if (cudaMallocHost((void **)&a->host_input, input_bytes) != cudaSuccess) goto fail;
    a->host_staging_bytes = input_bytes;
    a->x = a->input;
    a->decay = a->x + hidden;
    a->beta = a->decay + a->P;
    a->gate = a->beta + H;
    a->profile = getenv("GLM53_CUDA_PROFILE") &&
                 !strcmp(getenv("GLM53_CUDA_PROFILE"), "1");
    if (a->profile) for (int i = 0; i < 7; i++)
        if (cudaEventCreate(&a->mark[i]) != cudaSuccess) goto fail;
    if (cudaMemcpyAsync(a->conv, conv, (size_t)3 * a->P * K * 4,
                        cudaMemcpyHostToDevice, a->stream) != cudaSuccess ||
        cudaMemcpyAsync(a->norm, norm, (size_t)D * 4,
                        cudaMemcpyHostToDevice, a->stream) != cudaSuccess ||
        cudaStreamSynchronize(a->stream) != cudaSuccess) goto fail;
    return a;
fail:
    kda_free(a);
    return nullptr;
}

static bool kda_set_state(KdaProto *a, const float *state, const float *window) {
    if (cudaSetDevice(a->device) != cudaSuccess) return false;
    int next = 1 - a->committed;
    size_t sb = (size_t)a->H * a->D * a->D * 4;
    size_t wb = (size_t)3 * a->P * a->K * 4;
    if (cudaMemcpyAsync(a->state[next], state, sb,
                        cudaMemcpyHostToDevice, a->stream) != cudaSuccess ||
        cudaMemcpyAsync(a->window[next], window, wb,
                        cudaMemcpyHostToDevice, a->stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(a->stream) != cudaSuccess) return false;
    a->committed = next;
    return true;
}

static bool kda_get_state(KdaProto *a, float *state, float *window) {
    if (cudaSetDevice(a->device) != cudaSuccess) return false;
    size_t sb = (size_t)a->H * a->D * a->D * 4;
    size_t wb = (size_t)3 * a->P * a->K * 4;
    if (cudaMemcpyAsync(state, a->state[a->committed], sb,
                        cudaMemcpyDeviceToHost, a->stream) != cudaSuccess ||
        cudaMemcpyAsync(window, a->window[a->committed], wb,
                        cudaMemcpyDeviceToHost, a->stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(a->stream) == cudaSuccess;
}

__global__ static void kda_conv_proto(float *mixed, float *next_window,
                                     const float *window, const float *qkv,
                                     const float *taps, int channels, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    size_t base = (size_t)c * K;
    float sum = 0;
    for (int j = 0; j < K; j++) {
        float v = j == K - 1 ? qkv[c] : window[base + j + 1];
        next_window[base + j] = v;
        sum += taps[base + j] * v;
    }
    mixed[c] = sum / (1.0f + expf(-sum));
}

/* One output value per thread; each thread independently follows the CPU's
 * ascending key reduction. No head-scalar decay is representable here. */
__global__ static void kda_recur_proto(float *next, float *core,
                                      const float *state, const float *mixed,
                                      const float *decay, const float *beta,
                                      int H, int D) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (h >= H || v >= D) return;
    const float *q = mixed + h * D;
    const float *k = mixed + H * D + h * D;
    const float *value = mixed + 2 * H * D + h * D;
    float qs = 1e-6f, ks = 1e-6f;
    for (int j = 0; j < D; j++) { qs += q[j] * q[j]; ks += k[j] * k[j]; }
    float qnorm = (1.0f / sqrtf((float)D)) / sqrtf(qs);
    float knorm = 1.0f / sqrtf(ks);
    float prediction = 0;
    for (int j = 0; j < D; j++) {
        size_t off = ((size_t)h * D + j) * D + v;
        float s = state[off] * expf(decay[h * D + j]);
        next[off] = s;
        prediction += (k[j] * knorm) * s;
    }
    float answer = 0;
    for (int j = 0; j < D; j++) {
        size_t off = ((size_t)h * D + j) * D + v;
        float s = next[off] + (k[j] * knorm) *
                  (value[v] - prediction) * beta[h];
        next[off] = s;
        answer += (q[j] * qnorm) * s;
    }
    core[h * D + v] = answer;
}

__global__ static void kda_norm_proto(float *out, const float *core,
                                     const float *norm, const float *gate,
                                     int H, int D, float eps) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (h >= H || d >= D) return;
    float square = 0;
    for (int j = 0; j < D; j++) square += core[h * D + j] * core[h * D + j];
    float inv = 1.0f / sqrtf(square / D + eps);
    float g = gate[h * D + d];
    float sigmoid = g >= 0.0f ? 1.0f / (1.0f + expf(-g)) :
                               expf(g) / (1.0f + expf(g));
    out[h * D + d] = core[h * D + d] * inv * norm[d] *
                     sigmoid;
}

enum KdaFault { KDA_OK, KDA_FAIL_BEFORE, KDA_FAIL_AFTER_RECURRENCE };

static bool kda_step(KdaProto *a, float *out, const float *x,
                     const float *decay, const float *beta, const float *gate,
                     KdaFault fault = KDA_OK, float output_eps = 1e-6f) {
    if (fault == KDA_FAIL_BEFORE || cudaSetDevice(a->device) != cudaSuccess)
        return false;
    int next = 1 - a->committed;
    cudaStream_t s = a->stream;
    size_t input_bytes = ((size_t)a->hidden + 2u * a->P + a->H) * sizeof(float);
    memcpy(a->host_input, x, (size_t)a->hidden * 4);
    memcpy(a->host_input + a->hidden, decay, (size_t)a->P * 4);
    memcpy(a->host_input + a->hidden + a->P, beta, (size_t)a->H * 4);
    memcpy(a->host_input + a->hidden + a->P + a->H, gate, (size_t)a->P * 4);
    if (a->profile) cudaEventRecord(a->mark[0], s);
    if (cudaMemcpyAsync(a->input, a->host_input, input_bytes,
                        cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    if (a->profile) cudaEventRecord(a->mark[1], s);
    for (int j = 0; j < 3; j++) {
        ColiCudaTensor *w = a->proj[j];
        quant_matmul<<<dim3(a->P, 1), 256, 0, s>>>(
            a->qkv + j * a->P, a->x, w->weights, w->scales,
            4, 1, a->hidden, a->P, w->weight_bytes / a->P, 64, w->ng);
    }
    if (a->profile) cudaEventRecord(a->mark[2], s);
    kda_conv_proto<<<(3 * a->P + 255) / 256, 256, 0, s>>>(
        a->mixed, a->window[next], a->window[a->committed],
        a->qkv, a->conv, 3 * a->P, a->K);
    if (a->profile) cudaEventRecord(a->mark[3], s);
    kda_recur_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->state[next], a->core, a->state[a->committed], a->mixed,
        a->decay, a->beta, a->H, a->D);
    if (a->profile) cudaEventRecord(a->mark[4], s);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (fault == KDA_FAIL_AFTER_RECURRENCE) {
        cudaStreamSynchronize(s);
        return false;
    }
    kda_norm_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->normed, a->core, a->norm, a->gate, a->H, a->D, output_eps);
    ColiCudaTensor *w = a->proj[3];
    quant_matmul<<<dim3(a->hidden, 1), 256, 0, s>>>(
        a->out, a->normed, w->weights, w->scales,
        4, 1, a->P, a->hidden, w->weight_bytes / a->hidden, 64, w->ng);
    if (a->profile) cudaEventRecord(a->mark[5], s);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(out, a->out, (size_t)a->hidden * 4,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
    if (a->profile && cudaEventRecord(a->mark[6], s) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(s) != cudaSuccess) return false;
    if (a->profile) {
        double *fields[] = {&a->times.h2d_s, &a->times.projection_s,
            &a->times.convolution_s, &a->times.recurrence_s,
            &a->times.norm_output_s, &a->times.d2h_s};
        for (int i = 0; i < 6; i++) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, a->mark[i], a->mark[i + 1]) != cudaSuccess)
                return false;
            *fields[i] += (double)ms / 1000.0;
        }
    }
    a->committed = next;
    return true;
}

extern "C" ColiCudaKda *coli_cuda_kda_create(int device, int heads,
    int head_dim, int hidden, int kernel, const ColiCudaKdaMatrix matrices[4],
    const float *conv, const float *norm) {
    return kda_create(device, heads, head_dim, hidden, kernel, matrices, conv, norm);
}
extern "C" void coli_cuda_kda_free(ColiCudaKda *kda) { kda_free(kda); }
extern "C" int coli_cuda_kda_set_state(ColiCudaKda *kda,
    const float *state, const float *window) {
    return kda && state && window && kda_set_state(kda, state, window);
}
extern "C" int coli_cuda_kda_get_state(ColiCudaKda *kda,
    float *state, float *window) {
    return kda && state && window && kda_get_state(kda, state, window);
}
extern "C" int coli_cuda_kda_step(ColiCudaKda *kda, float *out,
    const float *x, const float *decay, const float *beta, const float *gate,
    float output_eps) {
    return kda && out && x && decay && beta && gate && output_eps > 0 &&
           kda_step(kda, out, x, decay, beta, gate, KDA_OK, output_eps);
}
extern "C" int coli_cuda_kda_footprint(const ColiCudaKda *kda,
    ColiCudaKdaFootprint *out) {
    if (!kda || !out) return 0;
    *out = {kda->projection_bytes, kda->state_bytes,
            kda->transactional_state_bytes, kda->window_bytes,
            kda->scratch_bytes, kda->other_bytes, kda->host_staging_bytes};
    return 1;
}
extern "C" int coli_cuda_kda_times(const ColiCudaKda *kda,
    ColiCudaKdaTimes *out) {
    if (!kda || !out) return 0;
    *out = kda->times;
    return 1;
}
