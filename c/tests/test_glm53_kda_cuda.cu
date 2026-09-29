/* Standalone Phase 2F1 prototype. Including the backend in this test TU gives
 * the object the exact production fmt-4/group-64 quant_matmul kernel without
 * exporting a production KDA API or changing glm53.c. */
#include "../backend_cuda.cu"
#include "../delta_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct KdaTestMat {
    int in, out;
    std::vector<uint8_t> q;
    std::vector<float> scale;
};

struct KdaProto {
    int device, H, D, P, hidden, K, committed;
    cudaStream_t stream;
    ColiCudaTensor *proj[4];
    float *state[2], *window[2];
    float *conv, *norm, *x, *decay, *beta, *gate;
    float *qkv, *mixed, *core, *normed, *out;
    size_t projection_bytes, state_bytes, transactional_state_bytes;
    size_t window_bytes, scratch_bytes, other_bytes;
};

static bool kda_alloc(float **p, size_t bytes, size_t *account) {
    *p = nullptr;
    if (cudaMalloc((void **)p, bytes) != cudaSuccess) return false;
    *account += coli_cuda_alloc_footprint(bytes);
    return true;
}

static void kda_free(KdaProto *a) {
    if (!a) return;
    if (a->stream) cudaStreamSynchronize(a->stream);
    for (int i = 0; i < 4; i++) coli_cuda_tensor_free(a->proj[i]);
    for (int i = 0; i < 2; i++) { cudaFree(a->state[i]); cudaFree(a->window[i]); }
    cudaFree(a->conv); cudaFree(a->norm); cudaFree(a->x);
    cudaFree(a->decay); cudaFree(a->beta); cudaFree(a->gate);
    cudaFree(a->qkv); cudaFree(a->mixed); cudaFree(a->core);
    cudaFree(a->normed); cudaFree(a->out);
    if (a->stream) cudaStreamDestroy(a->stream);
    delete a;
}

static KdaProto *kda_create(int device, int H, int D, int hidden, int K,
                            const KdaTestMat *mat, const float *conv,
                            const float *norm) {
    if (H < 2 || D < 2 || K != 4 || hidden % 64 || (H * D) % 64) return nullptr;
    KdaProto *a = new KdaProto{};
    a->device = device; a->H = H; a->D = D; a->P = H * D;
    a->hidden = hidden; a->K = K;
    if (cudaSetDevice(device) != cudaSuccess) goto fail;
    for (int i = 0; i < 4; i++) {
        if (mat[i].in != (i == 3 ? a->P : hidden) ||
            mat[i].out != (i == 3 ? hidden : a->P) ||
            !coli_cuda_tensor_upload_g(&a->proj[i], mat[i].q.data(),
                                       mat[i].scale.data(), 4, mat[i].in,
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
        !kda_alloc(&a->x, (size_t)hidden * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->decay, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->beta, (size_t)H * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->gate, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->qkv, (size_t)3 * a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->mixed, (size_t)3 * a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->core, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->normed, (size_t)a->P * 4, &a->scratch_bytes) ||
        !kda_alloc(&a->out, (size_t)hidden * 4, &a->scratch_bytes)) goto fail;
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
    out[h * D + d] = core[h * D + d] * inv * norm[d] *
                     (1.0f / (1.0f + expf(-g)));
}

enum KdaFault { KDA_OK, KDA_FAIL_BEFORE, KDA_FAIL_AFTER_RECURRENCE };

static bool kda_step(KdaProto *a, float *out, const float *x,
                     const float *decay, const float *beta, const float *gate,
                     KdaFault fault = KDA_OK) {
    if (fault == KDA_FAIL_BEFORE || cudaSetDevice(a->device) != cudaSuccess)
        return false;
    int next = 1 - a->committed;
    cudaStream_t s = a->stream;
    if (cudaMemcpyAsync(a->x, x, (size_t)a->hidden * 4,
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(a->decay, decay, (size_t)a->P * 4,
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(a->beta, beta, (size_t)a->H * 4,
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(a->gate, gate, (size_t)a->P * 4,
                        cudaMemcpyHostToDevice, s) != cudaSuccess) return false;
    for (int j = 0; j < 3; j++) {
        ColiCudaTensor *w = a->proj[j];
        quant_matmul<<<dim3(a->P, 1), 256, 0, s>>>(
            a->qkv + j * a->P, a->x, w->weights, w->scales,
            4, 1, a->hidden, a->P, w->weight_bytes / a->P, 64, w->ng);
    }
    kda_conv_proto<<<(3 * a->P + 255) / 256, 256, 0, s>>>(
        a->mixed, a->window[next], a->window[a->committed],
        a->qkv, a->conv, 3 * a->P, a->K);
    kda_recur_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->state[next], a->core, a->state[a->committed], a->mixed,
        a->decay, a->beta, a->H, a->D);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (fault == KDA_FAIL_AFTER_RECURRENCE) {
        cudaStreamSynchronize(s);
        return false;
    }
    kda_norm_proto<<<dim3((a->D + 127) / 128, a->H), 128, 0, s>>>(
        a->normed, a->core, a->norm, a->gate, a->H, a->D, 1e-6f);
    ColiCudaTensor *w = a->proj[3];
    quant_matmul<<<dim3(a->hidden, 1), 256, 0, s>>>(
        a->out, a->normed, w->weights, w->scales,
        4, 1, a->P, a->hidden, w->weight_bytes / a->hidden, 64, w->ng);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(out, a->out, (size_t)a->hidden * 4,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) return false;
    a->committed = next;
    return true;
}

static float cpu_gemv_row(const KdaTestMat &m, const float *x, int row) {
    /* Scalar branch of quant.h:matmul_i4_grouped_row, including its pairwise
     * accumulation and per-group scale application. */
    const uint8_t *w = m.q.data() + (size_t)row * (m.in / 2);
    const float *scl = m.scale.data() + (size_t)row * (m.in / 64);
    float sum = 0;
    for (int g = 0; g < m.in / 64; g++) {
        float sc = scl[g];
        for (int i = g * 64; i < (g + 1) * 64; i += 2) {
            uint8_t byte = w[i >> 1];
            sum += (x[i] * (float)((int)(byte & 15) - 8) +
                    x[i + 1] * (float)((int)(byte >> 4) - 8)) * sc;
        }
    }
    return sum;
}

static void cpu_step(int H, int D, int hidden, const KdaTestMat *mat,
                     const float *conv, const float *norm, float *state,
                     float *window, const float *x, const float *decay,
                     const float *beta, const float *gate, float *out) {
    int P = H * D;
    std::vector<float> qkv((size_t)3 * P), core(P);
    std::vector<float> scratch(coli_kda_scratch_floats(H, D, D));
    for (int j = 0; j < 3; j++)
        for (int r = 0; r < P; r++) qkv[j * P + r] = cpu_gemv_row(mat[j], x, r);
    coli_kda_step(core.data(), state, window, qkv.data(), conv, decay,
                  beta, H, D, D, 4, 1e-6f, scratch.data());
    std::vector<float> normed(P);
    for (int h = 0; h < H; h++) {
        float square = 0;
        for (int d = 0; d < D; d++) square += core[h * D + d] * core[h * D + d];
        float inv = 1.0f / sqrtf(square / D + 1e-6f);
        for (int d = 0; d < D; d++) {
            int i = h * D + d;
            normed[i] = core[i] * inv * norm[d] *
                        (1.0f / (1.0f + expf(-gate[i])));
        }
    }
    for (int r = 0; r < hidden; r++) out[r] = cpu_gemv_row(mat[3], normed.data(), r);
}

struct Error {
    float abs = 0, rel = 0;
    int token = -1, index = -1;
    void add(const float *ref, const float *got, size_t n, int t) {
        for (size_t i = 0; i < n; i++) {
            float d = fabsf(ref[i] - got[i]);
            float r = d / fmaxf(1e-4f, fabsf(ref[i]));
            if (d > abs) { abs = d; token = t; index = (int)i; }
            if (r > rel) rel = r;
        }
    }
};

static KdaTestMat make_mat(int in, int out, int seed) {
    KdaTestMat m{}; m.in = in; m.out = out;
    m.q.resize((size_t)out * in / 2);
    m.scale.resize((size_t)out * in / 64);
    for (int r = 0; r < out; r++) {
        for (int g = 0; g < in / 64; g++)
            m.scale[(size_t)r * (in / 64) + g] =
                0.002f * (1 + ((r + g + seed) % 7));
        for (int i = 0; i < in; i++) {
            int level = ((r * 13 + i * 7 + seed * 17) % 15) - 7;
            size_t at = (size_t)r * (in / 2) + i / 2;
            m.q[at] |= (uint8_t)((level + 8) << ((i & 1) * 4));
        }
    }
    return m;
}

static void inputs(int token, int H, int D, int hidden,
                   std::vector<float> &x, std::vector<float> &decay,
                   std::vector<float> &beta, std::vector<float> &gate) {
    for (int i = 0; i < hidden; i++)
        x[i] = 0.1f * sinf((float)(token * 11 + i * 3) * 0.13f);
    for (int h = 0; h < H; h++) {
        beta[h] = h & 1 ? 0.85f : 0.14f;
        for (int d = 0; d < D; d++) {
            decay[h * D + d] = -0.03f - 0.005f * (d % 13) - 0.02f * h;
            gate[h * D + d] = 0.07f * ((d % 11) - 5) + 0.1f * h;
        }
    }
}

static bool run_case(int H, int D, int hidden, int length, int initial,
                     int cpu_prefix, int cpu_suffix, KdaFault fault,
                     Error &oe, Error &se, Error &we, int reset_at = -1) {
    int P = H * D;
    KdaTestMat mat[4] = {make_mat(hidden, P, 1), make_mat(hidden, P, 2),
                         make_mat(hidden, P, 3), make_mat(P, hidden, 4)};
    std::vector<float> conv((size_t)3 * P * 4), norm(D);
    for (size_t i = 0; i < conv.size(); i++) conv[i] = 0.04f * sinf(i * 0.13f);
    for (int d = 0; d < D; d++) norm[d] = 0.7f + 0.003f * d;
    KdaProto *gpu = kda_create(0, H, D, hidden, 4, mat, conv.data(), norm.data());
    if (!gpu) { fprintf(stderr, "KDA create failed\n"); return false; }
    std::vector<float> ref((size_t)H * D * D), dev_state(ref.size()),
                       win((size_t)3 * P * 4), dev_win(win.size());
    for (size_t i = 0; i < ref.size(); i++) ref[i] = (initial & 1) ?
        0.02f * sinf(i * 0.017f) : 0;
    for (size_t i = 0; i < win.size(); i++) win[i] = (initial & 2) ?
        0.06f * cosf(i * 0.023f) : 0;
    if (!kda_set_state(gpu, ref.data(), win.data()) ||
        !kda_get_state(gpu, dev_state.data(), dev_win.data()) ||
        memcmp(ref.data(), dev_state.data(), ref.size() * 4) ||
        memcmp(win.data(), dev_win.data(), win.size() * 4)) {
        fprintf(stderr, "state round trip failed\n"); kda_free(gpu); return false;
    }
    std::vector<float> x(hidden), decay(P), beta(H), gate(P);
    std::vector<float> ro(hidden), go(hidden), prior_state, prior_win;
    std::vector<float> first_output(hidden);
    bool ok = true;
    for (int t = 0; t < length && ok; t++) {
        if (t == reset_at) {
            std::fill(ref.begin(), ref.end(), 0.0f);
            std::fill(win.begin(), win.end(), 0.0f);
            ok = kda_set_state(gpu, ref.data(), win.data());
            if (!ok) break;
        }
        inputs(reset_at >= 0 && t >= reset_at ? t - reset_at : t,
               H, D, hidden, x, decay, beta, gate);
        if (t == cpu_prefix) ok = kda_set_state(gpu, ref.data(), win.data());
        if (!ok) break;
        if (t < cpu_prefix || t >= length - cpu_suffix) {
            if (t == length - cpu_suffix) {
                ok = kda_get_state(gpu, dev_state.data(), dev_win.data());
                if (!ok) break;
                se.add(ref.data(), dev_state.data(), ref.size(), t);
                we.add(win.data(), dev_win.data(), win.size(), t);
            }
            cpu_step(H, D, hidden, mat, conv.data(), norm.data(),
                     ref.data(), win.data(), x.data(), decay.data(),
                     beta.data(), gate.data(), ro.data());
            if (t < cpu_prefix) continue;
            cpu_step(H, D, hidden, mat, conv.data(), norm.data(),
                     dev_state.data(), dev_win.data(), x.data(), decay.data(),
                     beta.data(), gate.data(), go.data());
        } else {
            if (fault != KDA_OK && t == cpu_prefix + 2) {
                prior_state = dev_state; prior_win = dev_win;
                ok = kda_get_state(gpu, prior_state.data(), prior_win.data()) &&
                     !kda_step(gpu, go.data(), x.data(), decay.data(),
                               beta.data(), gate.data(), fault) &&
                     kda_get_state(gpu, dev_state.data(), dev_win.data()) &&
                     !memcmp(prior_state.data(), dev_state.data(), ref.size() * 4) &&
                     !memcmp(prior_win.data(), dev_win.data(), win.size() * 4);
                if (!ok) break;
                if (fault == KDA_FAIL_AFTER_RECURRENCE) {
                    std::vector<float> next_state(ref.size()), next_window(win.size());
                    int next = 1 - gpu->committed;
                    ok = cudaMemcpyAsync(next_state.data(), gpu->state[next],
                                         ref.size() * 4, cudaMemcpyDeviceToHost,
                                         gpu->stream) == cudaSuccess &&
                         cudaMemcpyAsync(next_window.data(), gpu->window[next],
                                         win.size() * 4, cudaMemcpyDeviceToHost,
                                         gpu->stream) == cudaSuccess &&
                         cudaStreamSynchronize(gpu->stream) == cudaSuccess &&
                         memcmp(prior_state.data(), next_state.data(), ref.size() * 4) &&
                         memcmp(prior_win.data(), next_window.data(), win.size() * 4);
                    if (!ok) break;
                }
                /* CPU handles the same token from the committed generation. */
                cpu_step(H, D, hidden, mat, conv.data(), norm.data(),
                         dev_state.data(), dev_win.data(), x.data(), decay.data(),
                         beta.data(), gate.data(), go.data());
            } else {
                ok = kda_step(gpu, go.data(), x.data(), decay.data(),
                              beta.data(), gate.data());
                if (!ok) break;
            }
            cpu_step(H, D, hidden, mat, conv.data(), norm.data(),
                     ref.data(), win.data(), x.data(), decay.data(),
                     beta.data(), gate.data(), ro.data());
            if (fault == KDA_OK || t != cpu_prefix + 2)
                ok = kda_get_state(gpu, dev_state.data(), dev_win.data());
            else ok = kda_set_state(gpu, dev_state.data(), dev_win.data());
        }
        if (!ok) break;
        oe.add(ro.data(), go.data(), hidden, t);
        se.add(ref.data(), dev_state.data(), ref.size(), t);
        we.add(win.data(), dev_win.data(), win.size(), t);
        if (t == 0) first_output = go;
        if (t == reset_at && memcmp(first_output.data(), go.data(),
                                    (size_t)hidden * sizeof(float))) ok = false;
    }
    size_t total = gpu->projection_bytes + gpu->state_bytes +
                   gpu->transactional_state_bytes + gpu->window_bytes +
                   gpu->scratch_bytes + gpu->other_bytes;
    printf("footprint H=%d D=%d hidden=%d projection_bytes=%zu state_bytes=%zu transactional_state_bytes=%zu window_bytes=%zu scratch_bytes=%zu other_bytes=%zu total_bytes=%zu\n",
           H, D, hidden, gpu->projection_bytes, gpu->state_bytes,
           gpu->transactional_state_bytes, gpu->window_bytes,
           gpu->scratch_bytes, gpu->other_bytes, total);
    kda_free(gpu);
    return ok;
}

int main() {
    int device = 0;
    if (!coli_cuda_init(&device, 1)) { fprintf(stderr, "CUDA unavailable\n"); return 77; }
    /* Strict screening limits, not empirical acceptance tolerances. A real
     * GPU run must report the observed maxima before Phase 2F2 can proceed. */
    const float output_limit = 1e-3f, state_limit = 1e-4f;
    int failed = 0;
    const int lengths[] = {1, 2, 4, 8, 16};
    for (int n : lengths) for (int initial = 0; initial < 4; initial++) {
        Error o, s, w;
        bool ok = run_case(2, 64, 64, n, initial, 0, 0, KDA_OK, o, s, w);
        printf("sequence length=%d initial=%d pass=%d output_abs=%g output_rel=%g output_at=%d:%d state_abs=%g state_rel=%g state_at=%d:%d window_abs=%g window_rel=%g window_at=%d:%d\n",
               n, initial, ok, o.abs, o.rel, o.token, o.index,
               s.abs, s.rel, s.token, s.index, w.abs, w.rel, w.token, w.index);
        failed += !ok || o.abs > output_limit || s.abs > state_limit || w.abs > 1e-6f;
    }
    for (int mode = 0; mode < 6; mode++) {
        Error o, s, w;
        int prefix = mode == 0 || mode == 2 ? 3 : 0;
        int suffix = mode == 1 || mode == 2 ? 3 : 0;
        KdaFault fault = mode == 3 ? KDA_FAIL_AFTER_RECURRENCE :
                         mode == 4 ? KDA_FAIL_BEFORE : KDA_OK;
        bool ok = run_case(2, 64, 64, 8, mode != 5 ? 3 : 0, prefix, suffix,
                           fault, o, s, w, mode == 5 ? 4 : -1);
        printf("transition mode=%d pass=%d output_abs=%g output_rel=%g output_at=%d:%d state_abs=%g state_rel=%g state_at=%d:%d window_abs=%g window_rel=%g window_at=%d:%d\n",
               mode, ok, o.abs, o.rel, o.token, o.index,
               s.abs, s.rel, s.token, s.index, w.abs, w.rel, w.token, w.index);
        failed += !ok || o.abs > output_limit || s.abs > state_limit || w.abs > 1e-6f;
    }
    { Error o, s, w;
      bool ok = run_case(4, 128, 256, 4, 3, 0, 0, KDA_OK, o, s, w);
      printf("larger pass=%d output_abs=%g state_abs=%g window_abs=%g\n",
             ok, o.abs, s.abs, w.abs);
      failed += !ok || o.abs > output_limit || s.abs > state_limit || w.abs > 1e-6f;
    }
    coli_cuda_shutdown();
    return failed ? 1 : 0;
}
