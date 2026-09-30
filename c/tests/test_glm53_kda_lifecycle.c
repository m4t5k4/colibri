/* Production GSession/slot lifecycle with a transactional CPU-backed device
 * double. CUDA arithmetic/context correctness belongs to the CUDA proto test. */
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#define G53_CUDA_NO_TEST_MAIN
#define G53_CUDA_TEST_KDA_BACKEND
#include "test_glm53_cuda.c"

enum { LH = 2, LD = 4, LP = LH * LD, LX = 8,
       LS = LH * LD * LD, LW = 3 * LP * 4 };
struct KdaProto {
    GModel *model;
    float state[LS], window[LW], next_state[LS], next_window[LW];
    int pushes, pulls, steps, fail_push, fail_step;
};
typedef struct {
    GModel model;
    GLayer layer;
    G53KdaCudaLayer entry;
    ColiCudaKda device;
    float weights[9][LP * LX], conv[LW], norm[LD], alog[LH], dt[LP];
} LifecycleFixture;

static ColiCudaKda *startup_device;
static int startup_creates;

ColiCudaKda *coli_cuda_kda_create(int device, int heads, int hd, int hidden,
        int kernel, const ColiCudaKdaMatrix matrices[4], const float *conv,
        const float *norm) {
    (void)device; (void)heads; (void)hd; (void)hidden; (void)kernel;
    (void)matrices; (void)conv; (void)norm;
    assert(startup_device && "lifecycle fixtures attach an explicit device double");
    startup_creates++;
    return startup_device;
}
void coli_cuda_kda_free(ColiCudaKda *kda) { (void)kda; }
int coli_cuda_kda_set_state(ColiCudaKda *kda, const float *s, const float *w) {
    kda->pushes++;
    if (kda->fail_push) { kda->fail_push = 0; return 0; }
    memcpy(kda->state, s, sizeof(kda->state));
    memcpy(kda->window, w, sizeof(kda->window));
    return 1;
}
int coli_cuda_kda_get_state(ColiCudaKda *kda, float *s, float *w) {
    kda->pulls++;
    memcpy(s, kda->state, sizeof(kda->state));
    memcpy(w, kda->window, sizeof(kda->window));
    return 1;
}
int coli_cuda_kda_step(ColiCudaKda *kda, float *out, const float *x,
        const float *decay, const float *beta, const float *gate, float eps) {
    (void)decay; (void)beta; (void)gate; (void)eps;
    kda->steps++;
    memcpy(kda->next_state, kda->state, sizeof(kda->state));
    memcpy(kda->next_window, kda->window, sizeof(kda->window));
    GModel cpu = *kda->model;
    cpu.kda_tier.enabled = 0;
    GLayerState ls = {0};
    ls.kda_state = kda->next_state; ls.kda_window = kda->next_window;
    float scratch[256];
    assert(coli_kda_scratch_floats(LH, LD, LD) <= 256);
    GSession session = {0}; session.layer = &ls; session.kda_scratch = scratch;
    kda_layer(&cpu, &cpu.c, &cpu.layer[0], x, 1, out, &session, 0);
    if (kda->fail_step) { kda->fail_step = 0; return 0; }
    memcpy(kda->state, kda->next_state, sizeof(kda->state));
    memcpy(kda->window, kda->next_window, sizeof(kda->window));
    return 1;
}
int coli_cuda_kda_footprint(const ColiCudaKda *kda, ColiCudaKdaFootprint *out) {
    (void)kda; memset(out, 0, sizeof(*out)); return 1;
}
int coli_cuda_kda_times(const ColiCudaKda *kda, ColiCudaKdaTimes *out) {
    (void)kda; memset(out, 0, sizeof(*out)); return 1;
}

static void fixture(LifecycleFixture *f) {
    memset(f, 0, sizeof(*f));
    GModel *m = &f->model;
    m->c = (Cfg){.hidden=LX, .n_layers=1, .vocab=4, .kda_heads=LH,
        .kda_hd=LD, .kda_proj=LP, .conv_k=4, .gate_lb=-3.0f, .eps=1e-6f};
    m->layer = &f->layer; m->kda_gpu = &f->entry;
    m->kda_tier.enabled = 1; m->kda_tier.ndev = 1;
    f->entry.object = &f->device; f->entry.usable = 1;
    f->device.model = m;
    for (int a = 0; a < 9; a++) for (int i = 0; i < LP * LX; i++)
        f->weights[a][i] = (float)((i * 7 + a * 3) % 17 - 8) * 0.025f;
    Mat *matrices[] = {&f->layer.kq, &f->layer.kk, &f->layer.kv,
        &f->layer.ko, &f->layer.kfa, &f->layer.kfb, &f->layer.kb,
        &f->layer.kga, &f->layer.kgb};
    int rows[] = {LP,LP,LP,LX,LD,LP,LH,LD,LP};
    int cols[] = {LX,LX,LX,LP,LX,LD,LX,LX,LD};
    for (int a = 0; a < 9; a++)
        *matrices[a] = (Mat){.fmt=0, .f=f->weights[a], .rows=rows[a], .columns=cols[a]};
    for (int i = 0; i < LW; i++) f->conv[i] = 0.15f + (i % 7) * 0.03f;
    for (int i = 0; i < LD; i++) f->norm[i] = 0.8f + i * 0.1f;
    for (int i = 0; i < LH; i++) f->alog[i] = -0.4f + i * 0.2f;
    for (int i = 0; i < LP; i++) f->dt[i] = -0.3f + i * 0.06f;
    f->layer.conv=f->conv; f->layer.onorm=f->norm;
    f->layer.alog=f->alog; f->layer.dt=f->dt;
}
static void feature_gate(void) {
    const char *states[] = {NULL, "0", "1"};
    for (int i = 0; i < 3; i++) {
        LifecycleFixture f; fixture(&f);
        f.model.kda_gpu = NULL;
        memset(&f.model.kda_tier, 0, sizeof(f.model.kda_tier));
        Mat *projections[] = {&f.layer.kq, &f.layer.kk, &f.layer.kv, &f.layer.ko};
        for (int j = 0; j < 4; j++) {
            projections[j]->fmt = 4; projections[j]->gs = 64;
        }
        if (states[i]) setenv("GLM53_CUDA_KDA", states[i], 1);
        else unsetenv("GLM53_CUDA_KDA");
        setenv("COLI_CUDA", "1", 1); setenv("COLI_GPU", "0", 1);
        unsetenv("COLI_GPUS");
        startup_device = &f.device; startup_creates = init_calls = 0;
        glm53_kda_cuda_init(&f.model);
        int enabled = i != 1;
        assert(f.model.kda_tier.enabled == enabled);
        assert(init_calls == enabled && startup_creates == enabled);
        if (enabled) {
            assert(f.model.kda_gpu && f.model.kda_gpu[0].usable);
            assert(f.model.kda_gpu[0].object == &f.device);
            assert(!f.model.kda_gpu[0].valid && !f.model.kda_gpu[0].host_stale);
            assert(f.model.kda_tier.layers[0] == 1);
            glm53_kda_cuda_close(&f.model);
        } else assert(!f.model.kda_gpu);
        startup_device = NULL;
        printf("feature_gate=%s enabled=%d pass=1\n", states[i] ? states[i] : "UNSET", enabled);
    }
    /* Default ON must not bypass backend availability/capability guards. */
    unsetenv("GLM53_CUDA_KDA");
    GModel absent = {0}; init_calls = 0;
    setenv("COLI_CUDA", "0", 1);
    glm53_kda_cuda_init(&absent);
    assert(!absent.kda_tier.enabled && !absent.kda_gpu && init_calls == 0);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "invalid", 1);
    glm53_kda_cuda_init(&absent);
    assert(!absent.kda_tier.enabled && !absent.kda_gpu && init_calls == 0);
    unsetenv("COLI_GPUS");
    puts("feature_gate=UNSET_UNAVAILABLE enabled=0 pass=1");
}
static void equal(const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) assert(isfinite(a[i]) && isfinite(b[i]) &&
                                       fabsf(a[i] - b[i]) <= 1e-6f);
}
static void host_equal(GSession *a, GSession *b) {
    equal(a->layer[0].kda_state, b->layer[0].kda_state, LS);
    equal(a->layer[0].kda_window, b->layer[0].kda_window, LW);
}
static void token(LifecycleFixture *f, GSession *s, GSession *ref,
                  int index, int count, int decode) {
    float x[3 * LX], out[3 * LX], want[3 * LX];
    assert(count >= 1 && count <= 3);
    for (int i = 0; i < count * LX; i++)
        x[i] = (float)((i * 3 + index * 5) % 23 - 11) * 0.1f;
    GModel cpu = f->model; cpu.kda_tier.enabled = 0;
    kda_layer(&cpu, &cpu.c, &f->layer, x, count, want, ref, 0);
    f->model.cuda.decode_call = decode;
    kda_layer(&f->model, &f->model.c, &f->layer, x, count, out, s, 0);
    equal(out, want, count * LX);
    if (f->entry.owner == s && f->entry.host_stale) {
        assert(f->entry.valid);
        equal(f->device.state, ref->layer[0].kda_state, LS);
        equal(f->device.window, ref->layer[0].kda_window, LW);
    } else host_equal(s, ref);
}
static void alternating_sessions(void) {
    LifecycleFixture f; fixture(&f);
    GSession *a=session_open(&f.model,32), *b=session_open(&f.model,32);
    GSession *ar=session_open(&f.model,32), *br=session_open(&f.model,32);
    token(&f,a,ar,0,2,0); token(&f,b,br,7,2,0);
    for (int i=0; i<4; i++) {
        token(&f,a,ar,2+i,1,1);
        token(&f,a,ar,12+i,1,1);
        assert(f.entry.host_stale);
        token(&f,b,br,20+i,1,1);
        host_equal(a,ar); /* Switching materialized the outgoing session. */
        assert(f.entry.owner == b && f.entry.host_stale);
    }
    g53_kda_pull(&f.model,0); host_equal(b,br);
    assert(f.device.pushes == 8 && f.device.pulls == 8);
    session_close(&f.model,a); session_close(&f.model,b);
    assert(!f.entry.owner && !f.entry.valid && !f.entry.host_stale);
    session_close(&f.model,ar); session_close(&f.model,br);
    puts("lifecycle=ALTERNATING_SESSIONS pass=1");
}
static void prefill_transition(void) {
    LifecycleFixture f; fixture(&f);
    GSession *s=session_open(&f.model,32), *r=session_open(&f.model,32);
    token(&f,s,r,0,2,0); assert(f.device.steps == 0);
    token(&f,s,r,2,1,1); token(&f,s,r,3,1,1);
    assert(f.entry.host_stale && f.device.pulls == 0);
    token(&f,s,r,4,3,0);
    assert(!f.entry.valid && !f.entry.host_stale && f.device.pulls == 1);
    token(&f,s,r,7,1,1); assert(f.device.pushes == 2);
    token(&f,s,r,8,1,0); /* One-token prefill must also remain CPU. */
    assert(f.device.steps == 3 && f.device.pulls == 2 && !f.entry.valid);
    token(&f,s,r,9,1,1); assert(f.device.pushes == 3);
    session_close(&f.model,s); session_close(&f.model,r);
    puts("lifecycle=GPU_CPU_PREFILL_GPU pass=1");
}
static void pin_restore_reset(void) {
    LifecycleFixture f; fixture(&f);
    KVSlot slot={0}; slot.session=session_open(&f.model,32);
    GSession *r=session_open(&f.model,32), *saved=session_open(&f.model,32);
    int ids[]={1,2,3,4}; float logits[]={0,1,2,3};
    token(&f,slot.session,r,0,1,1); token(&f,slot.session,r,1,1,1);
    slot.session->filled=2; slot_remember(&slot,ids,2);
    assert(f.entry.host_stale && slot_pin_save(&f.model,&slot,ids,2,logits));
    assert(!f.entry.host_stale); host_equal(slot.session,r);
    memcpy(saved->layer[0].kda_state,r->layer[0].kda_state,LS*sizeof(float));
    memcpy(saved->layer[0].kda_window,r->layer[0].kda_window,LW*sizeof(float));
    token(&f,slot.session,r,2,1,1); token(&f,slot.session,r,3,1,1);
    slot.session->filled=4; slot_remember(&slot,ids,4);
    int pulls=f.device.pulls;
    assert(slot_pin_restore(&f.model,&slot,ids,3) == 2);
    assert(!f.entry.valid && !f.entry.host_stale && f.device.pulls == pulls);
    host_equal(slot.session,saved);
    token(&f,slot.session,saved,9,1,1); /* Push restored, not advanced state. */
    assert(f.device.pushes == 2);
    slot_reset(&f.model,&slot);
    assert(!slot.session && slot.n == 0 && !slot.pin_session);
    assert(!f.entry.owner && !f.entry.valid && !f.entry.host_stale);
    slot.session=session_open(&f.model,32);
    GSession *fresh=session_open(&f.model,32);
    token(&f,slot.session,fresh,0,1,1); assert(f.device.pushes == 3);
    slot_reset(&f.model,&slot); free(slot.tokens);
    /* Reset intentionally retains reusable pin ID/logit allocations. */
    for (int i=0; i<COLI_PIN_SLOTS_MAX; i++) {
        free(slot.pins.slot[i].ids); free(slot.pins.slot[i].logit);
    }
    session_close(&f.model,r); session_close(&f.model,saved); session_close(&f.model,fresh);
    puts("lifecycle=PIN_RESTORE_RESET pass=1");
}
static void failure_after_success(int push_failure) {
    LifecycleFixture f; fixture(&f);
    GSession *s=session_open(&f.model,32), *r=session_open(&f.model,32);
    token(&f,s,r,0,1,1); token(&f,s,r,1,1,1);
    assert(f.entry.host_stale && f.model.kda_tier.calls == 2);
    float committed[LS], window[LW];
    memcpy(committed,f.device.state,sizeof(committed));
    memcpy(window,f.device.window,sizeof(window));
    if (push_failure) {
        token(&f,s,r,2,2,0); /* Host becomes canonical before a new push. */
        f.device.fail_push=1;
    } else f.device.fail_step=1;
    token(&f,s,r,5,1,1);
    assert(!f.entry.usable && !f.entry.valid && !f.entry.host_stale);
    assert(f.model.kda_tier.errors == 1 && f.model.kda_tier.fallbacks == 1);
    assert(!memcmp(committed,f.device.state,sizeof(committed)) &&
           !memcmp(window,f.device.window,sizeof(window)));
    if (!push_failure) assert(memcmp(f.device.next_state,committed,sizeof(committed)));
    host_equal(s,r); /* Failed token processed once, from committed history. */
    int steps=f.device.steps, pushes=f.device.pushes;
    token(&f,s,r,6,1,1); token(&f,s,r,7,1,1);
    assert(f.device.steps == steps && f.device.pushes == pushes);
    assert(f.model.kda_tier.calls == 2 && f.model.kda_tier.fallbacks == 3);
    session_close(&f.model,s); session_close(&f.model,r);
    printf("lifecycle=%s pass=1 committed_unchanged=1 cpu_continuation=1\n",
           push_failure ? "PUSH_FAILURE_AFTER_SUCCESS" : "STEP_FAILURE_AFTER_SUCCESS");
}
int main(void) {
    setenv("COLI_PIN_SLOTS", "4", 1);
    feature_gate();
    alternating_sessions(); prefill_transition(); pin_restore_reset();
    failure_after_success(0); failure_after_success(1);
    puts("glm53 KDA production lifecycle: PASS (fake backend, no CUDA claim)");
    return 0;
}
