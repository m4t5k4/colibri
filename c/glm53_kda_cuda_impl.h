#include <new>

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

static bool kda_cuda_ok(cudaError_t err, const char *operation, int intended) {
    if (err == cudaSuccess) return true;
    int active = -1;
    (void)cudaGetDevice(&active);
    std::fprintf(stderr, "[glm53-kda-cuda-error] operation=%s intended_device=%d active_device=%d cuda_error=%d cuda_error_string=%s\n",
                 operation, intended, active, (int)err, cudaGetErrorString(err));
    /* A recoverable KDA error must not reach the expert launch error check. */
    (void)cudaGetLastError();
    return false;
}

/* Expert calls cache the calling thread's device in select_ctx(). KDA owns a
 * separate stream but must not leave that thread on its layer's GPU. Mark the
 * cache unknown after any direct switch so the next expert call reselects. */
struct KdaDeviceScope {
    int previous = -1, target;
    bool entered = false;
    explicit KdaDeviceScope(int device) : target(device) {
        if (kda_cuda_ok(cudaGetDevice(&previous), "get_device", target))
            entered = kda_cuda_ok(cudaSetDevice(target), "select_device", target);
        g_current_device = -1;
    }
    bool restore() {
        g_current_device = -1;
        if (entered) {
            entered = false;
            return kda_cuda_ok(cudaSetDevice(previous), "restore_device", previous);
        }
        return true;
    }
    ~KdaDeviceScope() {
        restore();
        g_current_device = -1;
    }
};

static bool kda_alloc(float **p, size_t bytes, size_t *account, int device) {
    *p = nullptr;
    if (!kda_cuda_ok(cudaMalloc((void **)p, bytes), "allocate", device)) return false;
    *account += coli_cuda_alloc_footprint(bytes);
    return true;
}

static void kda_free(KdaProto *a) {
    if (!a) return;
    KdaDeviceScope device(a->device);
    if (!device.entered) return;
    if (a->stream) kda_cuda_ok(cudaStreamSynchronize(a->stream), "free_wait", a->device);
    for (int i = 0; i < 4; i++) coli_cuda_tensor_free(a->proj[i]);
    auto release = [&](void *p) {
        if (p) kda_cuda_ok(cudaFree(p), "free_buffer", a->device);
    };
    for (int i = 0; i < 2; i++) { release(a->state[i]); release(a->window[i]); }
    release(a->conv); release(a->norm); release(a->input);
    if (a->host_input) kda_cuda_ok(cudaFreeHost(a->host_input), "free_host_staging", a->device);
    release(a->qkv); release(a->mixed); release(a->core);
    release(a->normed); release(a->out);
    for (int i = 0; i < 7; i++) if (a->mark[i])
        kda_cuda_ok(cudaEventDestroy(a->mark[i]), "free_profile_event", a->device);
    if (a->stream) kda_cuda_ok(cudaStreamDestroy(a->stream), "free_stream", a->device);
    kda_cuda_ok(cudaGetLastError(), "free_last_error", a->device);
    delete a;
}

static KdaProto *kda_create(int device, int H, int D, int hidden, int K,
                            const ColiCudaKdaMatrix *mat, const float *conv,
                            const float *norm) {
    if (H < 2 || D < 2 || K != 4 || hidden % 64 || (H * D) % 64) return nullptr;
    KdaDeviceScope selected(device);
    if (!selected.entered) return nullptr;
    KdaProto *a = new(std::nothrow) KdaProto{};
    if (!a) {
        std::fprintf(stderr, "[glm53-kda-cuda-error] operation=host_object_allocation intended_device=%d\n", device);
        return nullptr;
    }
    a->device = device; a->H = H; a->D = D; a->P = H * D;
    a->hidden = hidden; a->K = K;
    size_t input_bytes = ((size_t)hidden + 2u * a->P + H) * sizeof(float);
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
    if (!kda_cuda_ok(cudaStreamSynchronize(0), "projection_upload_wait", device) ||
        !kda_cuda_ok(cudaStreamCreateWithFlags(&a->stream, cudaStreamNonBlocking), "stream_create", device))
        goto fail;
    if (!kda_alloc(&a->state[0], (size_t)H * D * D * 4, &a->state_bytes, device) ||
        !kda_alloc(&a->state[1], (size_t)H * D * D * 4, &a->transactional_state_bytes, device) ||
        !kda_alloc(&a->window[0], (size_t)3 * a->P * K * 4, &a->window_bytes, device) ||
        !kda_alloc(&a->window[1], (size_t)3 * a->P * K * 4,
                   &a->transactional_state_bytes, device) ||
        !kda_alloc(&a->conv, (size_t)3 * a->P * K * 4, &a->other_bytes, device) ||
        !kda_alloc(&a->norm, (size_t)D * 4, &a->other_bytes, device) ||
        !kda_alloc(&a->input, input_bytes, &a->scratch_bytes, device) ||
        !kda_alloc(&a->qkv, (size_t)3 * a->P * 4, &a->scratch_bytes, device) ||
        !kda_alloc(&a->mixed, (size_t)3 * a->P * 4, &a->scratch_bytes, device) ||
        !kda_alloc(&a->core, (size_t)a->P * 4, &a->scratch_bytes, device) ||
        !kda_alloc(&a->normed, (size_t)a->P * 4, &a->scratch_bytes, device) ||
        !kda_alloc(&a->out, (size_t)hidden * 4, &a->scratch_bytes, device)) goto fail;
    if (!kda_cuda_ok(cudaMallocHost((void **)&a->host_input, input_bytes), "host_staging_allocate", device)) goto fail;
    a->host_staging_bytes = input_bytes;
    a->x = a->input;
    a->decay = a->x + hidden;
    a->beta = a->decay + a->P;
    a->gate = a->beta + H;
    a->profile = getenv("GLM53_CUDA_PROFILE") &&
                 !strcmp(getenv("GLM53_CUDA_PROFILE"), "1");
    if (a->profile) for (int i = 0; i < 7; i++)
        if (!kda_cuda_ok(cudaEventCreate(&a->mark[i]), "profile_event_create", device)) goto fail;
    if (!kda_cuda_ok(cudaMemcpyAsync(a->conv, conv, (size_t)3 * a->P * K * 4,
                        cudaMemcpyHostToDevice, a->stream), "convolution_upload", device) ||
        !kda_cuda_ok(cudaMemcpyAsync(a->norm, norm, (size_t)D * 4,
                        cudaMemcpyHostToDevice, a->stream), "norm_upload", device) ||
        !kda_cuda_ok(cudaStreamSynchronize(a->stream), "create_wait", device)) goto fail;
    if (!selected.restore()) goto fail;
    return a;
fail:
    kda_free(a);
    return nullptr;
}

static bool kda_set_state(KdaProto *a, const float *state, const float *window) {
    KdaDeviceScope device(a->device);
    if (!device.entered) return false;
    int next = 1 - a->committed;
    size_t sb = (size_t)a->H * a->D * a->D * 4;
    size_t wb = (size_t)3 * a->P * a->K * 4;
    if (!kda_cuda_ok(cudaMemcpyAsync(a->state[next], state, sb,
                        cudaMemcpyHostToDevice, a->stream), "state_push", a->device) ||
        !kda_cuda_ok(cudaMemcpyAsync(a->window[next], window, wb,
                        cudaMemcpyHostToDevice, a->stream), "window_push", a->device)) return false;
    if (!kda_cuda_ok(cudaStreamSynchronize(a->stream), "state_push_wait", a->device)) return false;
    if (!device.restore()) return false;
    a->committed = next;
    return true;
}

static bool kda_get_state(KdaProto *a, float *state, float *window) {
    KdaDeviceScope device(a->device);
    if (!device.entered) return false;
    size_t sb = (size_t)a->H * a->D * a->D * 4;
    size_t wb = (size_t)3 * a->P * a->K * 4;
    if (!kda_cuda_ok(cudaMemcpyAsync(state, a->state[a->committed], sb,
                        cudaMemcpyDeviceToHost, a->stream), "state_pull", a->device) ||
        !kda_cuda_ok(cudaMemcpyAsync(window, a->window[a->committed], wb,
                        cudaMemcpyDeviceToHost, a->stream), "window_pull", a->device)) return false;
    return kda_cuda_ok(cudaStreamSynchronize(a->stream), "state_pull_wait", a->device) &&
           device.restore();
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

enum KdaFault { KDA_OK, KDA_FAIL_BEFORE, KDA_FAIL_AFTER_RECURRENCE,
               KDA_FAIL_RUNTIME_AFTER_RECURRENCE, KDA_FAIL_RESTORE_BEFORE_COMMIT };

static bool kda_record(KdaProto *a, int index) {
    return !a->profile || kda_cuda_ok(cudaEventRecord(a->mark[index], a->stream),
                                     "profile_event_record", a->device);
}

static bool kda_step(KdaProto *a, float *out, const float *x,
                     const float *decay, const float *beta, const float *gate,
                     KdaFault fault = KDA_OK, float output_eps = 1e-6f) {
    if (fault == KDA_FAIL_BEFORE) return false;
    KdaDeviceScope device(a->device);
    if (!device.entered) return false;
    int next = 1 - a->committed;
    cudaStream_t s = a->stream;
    size_t input_bytes = ((size_t)a->hidden + 2u * a->P + a->H) * sizeof(float);
    memcpy(a->host_input, x, (size_t)a->hidden * 4);
    memcpy(a->host_input + a->hidden, decay, (size_t)a->P * 4);
    memcpy(a->host_input + a->hidden + a->P, beta, (size_t)a->H * 4);
    memcpy(a->host_input + a->hidden + a->P + a->H, gate, (size_t)a->P * 4);
    if (!kda_record(a, 0)) return false;
    if (!kda_cuda_ok(cudaMemcpyAsync(a->input, a->host_input, input_bytes,
                        cudaMemcpyHostToDevice, s), "token_upload", a->device)) return false;
    if (!kda_record(a, 1)) return false;
    for (int j = 0; j < 3; j++) {
        ColiCudaTensor *w = a->proj[j];
        quant_matmul<<<dim3(a->P, 1), 256, 0, s>>>(
            a->qkv + j * a->P, a->x, w->weights, w->scales,
            4, 1, a->hidden, a->P, w->weight_bytes / a->P, 64, w->ng);
    }
    if (!kda_record(a, 2)) return false;
    kda_conv_proto<<<(3 * a->P + 255) / 256, 256, 0, s>>>(
        a->mixed, a->window[next], a->window[a->committed],
        a->qkv, a->conv, 3 * a->P, a->K);
    if (!kda_record(a, 3)) return false;
    kda_recur_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->state[next], a->core, a->state[a->committed], a->mixed,
        a->decay, a->beta, a->H, a->D);
    if (!kda_record(a, 4)) return false;
    if (!kda_cuda_ok(cudaGetLastError(), "qkv_conv_recurrence_launch", a->device)) return false;
    if (fault == KDA_FAIL_AFTER_RECURRENCE || fault == KDA_FAIL_RUNTIME_AFTER_RECURRENCE) {
        if (!kda_cuda_ok(cudaStreamSynchronize(s), "injected_failure_wait", a->device)) return false;
        if (fault == KDA_FAIL_RUNTIME_AFTER_RECURRENCE)
            return kda_cuda_ok(cudaSetDevice(-1), "injected_invalid_device", a->device);
        return false;
    }
    kda_norm_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->normed, a->core, a->norm, a->gate, a->H, a->D, output_eps);
    ColiCudaTensor *w = a->proj[3];
    quant_matmul<<<dim3(a->hidden, 1), 256, 0, s>>>(
        a->out, a->normed, w->weights, w->scales,
        4, 1, a->P, a->hidden, w->weight_bytes / a->hidden, 64, w->ng);
    if (!kda_record(a, 5)) return false;
    if (!kda_cuda_ok(cudaGetLastError(), "norm_output_launch", a->device) ||
        !kda_cuda_ok(cudaMemcpyAsync(out, a->out, (size_t)a->hidden * 4,
                        cudaMemcpyDeviceToHost, s), "token_download", a->device)) return false;
    if (!kda_record(a, 6)) return false;
    if (!kda_cuda_ok(cudaStreamSynchronize(s), "token_wait", a->device)) return false;
    if (a->profile) {
        double *fields[] = {&a->times.h2d_s, &a->times.projection_s,
            &a->times.convolution_s, &a->times.recurrence_s,
            &a->times.norm_output_s, &a->times.d2h_s};
        for (int i = 0; i < 6; i++) {
            float ms = 0;
            if (!kda_cuda_ok(cudaEventElapsedTime(&ms, a->mark[i], a->mark[i + 1]),
                             "profile_event_elapsed", a->device))
                return false;
            *fields[i] += (double)ms / 1000.0;
        }
    }
    if (fault == KDA_FAIL_RESTORE_BEFORE_COMMIT) device.previous = -1;
    if (!device.restore()) return false;
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
