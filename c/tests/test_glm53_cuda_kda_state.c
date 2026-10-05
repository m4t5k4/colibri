/* Fake device callbacks; production session, coherence, runtime and lease code. */
#define _POSIX_C_SOURCE 200809L
#define COLI_CUDA
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#include "../glm53.c"
#include "../segment_runtime.c"
#include "../backend_cuda_lifetime.h"
#include <assert.h>
#include "cuda_kda_recur_ref.h"
#include "cuda_kda_shortconv_ref.h"

static ColiCudaLifetime lifetime;
static int inits, shutdowns, allocs, frees, uploads, downloads, acquires;
static int fail_alloc_at, fail_upload_at, fail_download_at, live;
static int recurrence_calls, shortconv_calls, sync_calls, fail_recurrence, fail_shortconv, fail_sync;
static int fail_recurrence_at, synthetic_numeric;
static size_t upload_bytes, download_bytes;
static Glm53SegmentEngine *watch_engine;
static float reference_state[18], reference_window[72], reference_core[6];
static GSession *watch_session;
static ColiGlm53CudaKdaLayer *watch_layer;
typedef struct { void *p; size_t bytes; int owner; } Buffer;
static Buffer buffers[128];
typedef struct { char op; int owner; const void *p; size_t bytes; } Event;
static Event events[8192];
static size_t nevents;
static void event(char op, int owner, const void *p, size_t bytes) {
    assert(nevents < sizeof(events)/sizeof(*events));
    events[nevents++] = (Event){op, owner, p, bytes};
}
static int init(const int *devices, int count) {
    assert(devices && count); inits++; return 1;
}
static void shutdown_backend(void) { assert(!live); shutdowns++; event('S', -1, NULL, 0); }
int coli_cuda_acquire(const int *devices, int count) {
    acquires++; event('A', -1, NULL, (size_t)count);
    return coli_cuda_lifetime_acquire(&lifetime, devices, count, init);
}
void coli_cuda_release(void) {
    event('R', -1, NULL, 0); coli_cuda_lifetime_release(&lifetime, shutdown_backend);
}
int coli_cuda_device_count(void) { return lifetime.users ? lifetime.count : 0; }
int coli_cuda_device_at(int i) { return i >= 0 && i < coli_cuda_device_count() ? lifetime.devices[i] : -1; }
static Buffer *find(int owner, const void *p, size_t bytes) {
    for (size_t i = 0; i < 128; i++) if (buffers[i].p) {
        uintptr_t start=(uintptr_t)buffers[i].p, at=(uintptr_t)p;
        if (at >= start && at-start <= buffers[i].bytes) {
            assert(lifetime.users && buffers[i].owner == owner && bytes <= buffers[i].bytes-(at-start));
            return &buffers[i];
        }
    }
    assert(0); return NULL;
}
void *coli_cuda_pipe_alloc(int owner, size_t bytes) {
    assert(lifetime.users && bytes);
    int found = 0;
    for (int i = 0; i < lifetime.count; i++) found |= lifetime.devices[i] == owner;
    assert(found);
    allocs++; event('N', owner, NULL, bytes);
    if (allocs == fail_alloc_at) return NULL;
    for (size_t i = 0; i < 128; i++) if (!buffers[i].p) {
        buffers[i] = (Buffer){malloc(bytes), bytes, owner}; assert(buffers[i].p);
        memset(buffers[i].p, 0xa5, bytes); live++; return buffers[i].p;
    }
    assert(0); return NULL;
}
void coli_cuda_pipe_free(int owner, void *p) {
    Buffer *b = find(owner, p, 0); event('F', owner, p, b->bytes);
    free(p); memset(b, 0, sizeof(*b)); live--; frees++;
}
int coli_cuda_pipe_upload(int owner, void *dst, const void *src, size_t bytes) {
    find(owner, dst, bytes); uploads++; upload_bytes+=bytes; event('U', owner, dst, bytes);
    if (watch_session) for (int i=0; i<32; i++)
        if (dst==watch_session->layer[i].cuda_kda.state || dst==watch_session->layer[i].cuda_kda.window)
            assert(watch_session->layer[i].cuda_kda.authority==G53_KDA_HOST);
    if (uploads == fail_upload_at) return 0;
    memcpy(dst, src, bytes); return 1;
}
int coli_cuda_pipe_download(int owner, const void *src, void *dst, size_t bytes) {
    find(owner, src, bytes); downloads++; download_bytes+=bytes; event('D', owner, src, bytes);
    if (watch_session && watch_layer && src == watch_session->kda_staging.raw_out)
        assert(watch_layer->authority == G53_KDA_UNKNOWN);
    if (downloads == fail_download_at) return 0;
    memcpy(dst, src, bytes); return 1;
}

int coli_cuda_pipe_kda_shortconv(int owner, float *window, float *mixed,
        const float *qkv, const float *conv, int channels, int kernel) {
    shortconv_calls++; event('C',owner,window,(size_t)channels*kernel*sizeof(float));
    find(owner,window,(size_t)channels*kernel*sizeof(float));
    find(owner,conv,(size_t)channels*kernel*sizeof(float));
    find(owner,mixed,(size_t)channels*sizeof(float));find(owner,qkv,(size_t)channels*sizeof(float));
    watch_layer=NULL;
    if (watch_session) for (int i=0; i<32; i++) if (watch_session->layer[i].cuda_kda.window==window) {
        watch_layer=&watch_session->layer[i].cuda_kda;
        assert(watch_engine && watch_engine->model.layer[i].cuda_kda_weights.conv==conv);
        assert(watch_layer->authority==G53_KDA_BOTH || watch_layer->authority==G53_KDA_DEVICE);
        assert(mixed==watch_session->kda_staging.mixed_qkv && qkv==watch_session->kda_staging.projected_qkv);
    }
    if (fail_shortconv) return 0;
    if (watch_session) {
        assert(watch_layer && channels==18 && kernel==4);
        memcpy(reference_state,watch_layer->state,72);memcpy(reference_window,window,288);
        int n=coli_kda_scratch_floats(2,3,3);float scratch[n];
        assert(!coli_kda_step(reference_core,reference_state,reference_window,qkv,conv,
            watch_session->kda_staging.decay,watch_session->kda_staging.beta,2,3,3,kernel,1e-6f,scratch));
    }
    kda_shortconv_ref(window,mixed,qkv,conv,channels,kernel);
    if (watch_session) assert(!memcmp(window,reference_window,288));
    return 1;
}
int coli_cuda_pipe_kda_recur(int owner, float *st, float *out,
        const float *q, const float *k, const float *v, const float *decay,
        const float *beta, int h, int kd, int vd, float eps) {
    recurrence_calls++; event('K', owner, st, (size_t)h*kd*vd*sizeof(float));
    find(owner,st,(size_t)h*kd*vd*sizeof(float)); find(owner,out,(size_t)h*vd*sizeof(float));
    find(owner,q,(size_t)h*kd*sizeof(float)); find(owner,k,(size_t)h*kd*sizeof(float));
    find(owner,v,(size_t)h*vd*sizeof(float)); find(owner,decay,(size_t)h*kd*sizeof(float));
    find(owner,beta,(size_t)h*sizeof(float));
    if (watch_session) {
        assert(watch_layer && watch_layer->state==st && watch_layer->authority==G53_KDA_UNKNOWN);
        assert(q==watch_session->kda_staging.mixed_qkv && k==q+(size_t)h*kd && v==q+2*(size_t)h*kd);
    }
    if (fail_recurrence || recurrence_calls==fail_recurrence_at) return 0;
    kda_recur_ref(st,out,q,k,v,decay,beta,h,kd,vd,eps);
    if (watch_session) {
        for (int i=0;i<18;i++) assert(fabsf(st[i]-reference_state[i])<=1e-5f*(1.f+fabsf(reference_state[i])));
        for (int i=0;i<6;i++) assert(fabsf(out[i]-reference_core[i])<=1e-5f*(1.f+fabsf(reference_core[i])));
    }
    return 1;
}
int coli_cuda_pipe_sync(int owner) {
    sync_calls++; event('Y',owner,NULL,0);
    if (watch_session) assert(watch_layer && watch_layer->authority==G53_KDA_UNKNOWN);
    return !fail_sync;
}

/* Only model loading is replaced: all lifecycle/state callbacks are real. */
static int synthetic_open(void **impl, ColiSegmentCapabilities *caps,
        const ColiSegmentEngineOptions *opts, char *error, size_t error_size) {
    (void)error; (void)error_size;
    Glm53SegmentEngine *e = calloc(1, sizeof(*e)); assert(e);
    e->model.c = (Cfg){.n_layers=32, .hidden=2, .hc_mult=1, .kda_heads=2,
        .kda_hd=3, .kda_proj=6, .conv_k=4, .kv_lora=2, .index_hd=2};
    for (int i = 0; i < 32; i++) e->model.c.is_full[i] = i % 3 == 0;
    e->model.layer_begin = e->layer_begin = opts->layer_begin;
    e->model.layer_end = e->layer_end = opts->layer_end;
    e->context_tokens = opts->context_tokens; e->state_width = 2;
    if (coli_glm53_cuda_stage_open(&e->cuda_stage, opts->resource_plan, opts->resource_plan_size) < 0) {
        free(e); return -1;
    }
    if (e->cuda_stage.lease_live) assert(coli_glm53_cuda_stage_wire_create(&e->cuda_stage, 1, 2) == 1);
    e->model.layer=calloc(32,sizeof(GLayer));assert(e->model.layer);
    for (int i=e->model.layer_begin; i<e->model.layer_end; i++) if (!e->model.c.is_full[i]) {
        e->model.layer[i].conv=calloc(72,sizeof(float));assert(e->model.layer[i].conv);
        if (synthetic_numeric) for (int j=0; j<72; j++)
            ((float *)e->model.layer[i].conv)[j]=(float)((j*3+i)%11-5)*0.09f;
    }
    if (!glm53_kda_weights_open(&e->model,&e->cuda_stage)) {
        model_release(&e->model);coli_glm53_cuda_stage_close(&e->cuda_stage);free(e);return -1;
    }
    assert(!pthread_mutex_init(&e->run_lock, NULL));
    *caps = (ColiSegmentCapabilities){.struct_size=sizeof(*caps), .abi_version=COLI_SEGMENT_ABI_VERSION,
        .flags=COLI_SEGMENT_CAP_CPU | COLI_SEGMENT_CAP_MULTI_SESSION | COLI_SEGMENT_CAP_RANGE_NATIVE | COLI_SEGMENT_CAP_SNAPSHOT,
        .state_dtype=COLI_SEGMENT_DTYPE_F32, .state_width=2, .num_layers=32,
        .max_context_tokens=8, .max_batch_rows=8};
    strcpy(caps->engine_id, "kda-test"); strcpy(caps->state_schema, "test-f32");
    strcpy(caps->numeric_class, "cpu-f32"); *impl = e; return 0;
}
static const ColiSegmentAdapter adapter = {
    sizeof(adapter), COLI_SEGMENT_ABI_VERSION, "kda-test", synthetic_open,
    glm53_segment_engine_destroy, glm53_segment_session_create, glm53_segment_session_destroy,
    glm53_segment_session_run, glm53_segment_session_snapshot, glm53_segment_session_restore, {0}
};
static ColiSegmentEngine *engine_open(int begin, int end, int owner) {
    ColiGlm53StagePlan plan = {sizeof(plan), COLI_GLM53_STAGE_PLAN_VERSION, owner};
    ColiSegmentEngineOptions opts = {.struct_size=sizeof(opts), .model_dir="synthetic",
        .layer_begin=(uint32_t)begin, .layer_end=(uint32_t)end, .context_tokens=8,
        .resource_plan=owner < 0 ? NULL : &plan, .resource_plan_size=owner < 0 ? 0 : sizeof(plan)};
    ColiSegmentEngine *e = NULL; char error[128];
    assert(!coli_segment_engine_open("kda-test", &opts, &e, error, sizeof(error))); return e;
}
static ColiSegmentSession *create(ColiSegmentEngine *e) {
    ColiSegmentSession *s = NULL; char error[128];
    ColiSegmentSessionOptions opts={.struct_size=sizeof(opts), .context_tokens=8};
    assert(!coli_segment_session_create(e, &opts, &s, error, sizeof(error))); return s;
}
static GSession *state(ColiSegmentSession *s) { return ((Glm53SegmentSession *)s->impl)->session; }
static void fill(float *p, size_t bytes, float value) {
    for (size_t i=0; i<bytes/sizeof(float); i++) p[i] = value + (float)i;
}
/* Test-only external device mutation seam for coherence/resource probes. */
static void device_write(GSession *s, int i, float value) {
    GLayerState *st = &s->layer[i]; ColiGlm53CudaKdaLayer *d = &st->cuda_kda;
    fill(d->state, d->state_bytes, value); fill(d->window, d->window_bytes, value+100);
    d->authority = G53_KDA_DEVICE;
}
typedef struct { unsigned char bytes[32768]; size_t size, offset; Glm53SegmentEngine *engine; } Stream;
static int write_stream(void *arg, const void *data, size_t bytes) {
    Stream *s=arg;
    /* Callback observes the existing lock held, without trying to re-enter. */
    assert(pthread_mutex_trylock(&s->engine->run_lock) == EBUSY);
    assert(bytes <= sizeof(s->bytes)-s->size); memcpy(s->bytes+s->size, data, bytes); s->size+=bytes; return 0;
}
static int read_stream(void *arg, void *data, size_t bytes) {
    Stream *s=arg; assert(pthread_mutex_trylock(&s->engine->run_lock) == EBUSY);
    if (bytes > s->size-s->offset) return -1;
    memcpy(data, s->bytes+s->offset, bytes); s->offset+=bytes; return 0;
}
static void range_check(int begin, int end, int owner) {
    ColiSegmentEngine *e=engine_open(begin,end,owner);
    int leases=acquires, before=allocs;
    ColiSegmentSession *a=create(e), *b=create(e); GSession *sa=state(a), *sb=state(b);
    Glm53SegmentEngine *ge=e->impl; int count=0;
    for (int i=0; i<32; i++) {
        int owned=i>=begin && i<end && !ge->model.c.is_full[i];
        ColiGlm53CudaKdaLayer *da=&sa->layer[i].cuda_kda, *db=&sb->layer[i].cuda_kda;
        assert(!!da->state == (owned && owner>=0)); assert(!!da->window == !!da->state);
        if (!da->state) continue;
        count++; assert(da->state != db->state && da->window != db->window);
        assert(da->state_bytes == 72 && da->window_bytes == 288 && da->authority == G53_KDA_HOST);
        assert(find(owner, da->state, 72)); assert(find(owner, da->window, 288));
        fill(sa->layer[i].kda_state,72,10); fill(sa->layer[i].kda_window,288,20);
        fill(sb->layer[i].kda_state,72,30); fill(sb->layer[i].kda_window,288,40);
        int u=uploads,d=downloads;
        assert(coli_glm53_cuda_kda_prepare(sa->cuda_stage,da,sa->layer[i].kda_state,sa->layer[i].kda_window));
        assert(uploads == u+2 && downloads == d);
        assert(coli_glm53_cuda_kda_prepare(sa->cuda_stage,da,sa->layer[i].kda_state,sa->layer[i].kda_window));
        assert(coli_glm53_cuda_kda_ensure_host(sa->cuda_stage,da,sa->layer[i].kda_state,sa->layer[i].kda_window));
        assert(uploads == u+2 && downloads == d);
        assert(coli_glm53_cuda_kda_prepare(sb->cuda_stage,db,sb->layer[i].kda_state,sb->layer[i].kda_window));
        device_write(sa,i,50);
        assert(coli_glm53_cuda_kda_prepare(sa->cuda_stage,da,sa->layer[i].kda_state,sa->layer[i].kda_window));
        assert(uploads == u+4); /* DEVICE prepare is a no-op */
        assert(glm53_kda_ensure_host(&ge->model,sa)); assert(downloads == d+2);
        assert(sa->layer[i].kda_state[0] == 50 && sa->layer[i].kda_window[0] == 150);
        assert(sb->layer[i].kda_state[0] == 30 && ((float *)db->state)[0] == 30);
    }
    assert(allocs == before+4*count+2*(count>0 && owner>=0) && acquires == leases);
    char error[128]; assert(coli_segment_engine_close(e,error,sizeof(error)) != 0);
    Stream snapshot={.engine=ge};
    assert(!coli_segment_snapshot(a,write_stream,&snapshot,error,sizeof(error)));
    for (int i=begin; i<end; i++) if (sb->layer[i].cuda_kda.state)
        assert(sb->layer[i].kda_state[0]==30 && ((float *)sb->layer[i].cuda_kda.state)[0]==30);
    int reverse=begin==7; GSession *survivor=reverse ? sa : sb;
    coli_segment_session_destroy(reverse ? b : a);
    for (int i=begin; i<end; i++) if (survivor->layer[i].cuda_kda.state) {
        device_write(survivor,i,70); assert(glm53_kda_ensure_host(&ge->model,survivor));
        assert(survivor->layer[i].kda_state[0] == 70);
    }
    coli_segment_session_destroy(reverse ? a : b); assert(!coli_segment_engine_close(e,error,sizeof(error)));
    printf("range [%d,%d), owner %d: %d KDA layers, independent sessions and ordered teardown: PASS\n",begin,end,owner,count);
}
static void failures(void) {
    ColiSegmentEngine *e=engine_open(4,8,4); Glm53SegmentEngine *ge=e->impl;
    ColiSegmentSession *a=create(e); GSession *s=state(a); GLayerState *st=&s->layer[4];
    int existing=live, leases=acquires;
    for (int n=1; n<=7; n++) { /* every layer allocation and shared staging block */
        fail_alloc_at=allocs+n; ColiSegmentSession *bad=NULL; char error[128];
        ColiSegmentSessionOptions opts={.struct_size=sizeof(opts), .context_tokens=8};
        assert(coli_segment_session_create(e,&opts,&bad,error,sizeof(error)) != 0 && !bad);
        assert(live==existing && e->active_sessions==1 && acquires==leases && lifetime.users==1);
    }
    fail_alloc_at=0;
    for (int n=1; n<=2; n++) {
        st->cuda_kda.authority=G53_KDA_HOST; fail_upload_at=uploads+n;
        assert(!coli_glm53_cuda_kda_prepare(s->cuda_stage,&st->cuda_kda,st->kda_state,st->kda_window));
        assert(st->cuda_kda.authority==G53_KDA_HOST);
        fail_upload_at=0;
        assert(coli_glm53_cuda_kda_prepare(s->cuda_stage,&st->cuda_kda,st->kda_state,st->kda_window));
    }
    for (int n=1; n<=2; n++) {
        fill(st->kda_state,72,11); fill(st->kda_window,288,22); device_write(s,4,99);
        fail_download_at=downloads+n;
        assert(!glm53_kda_ensure_host(&ge->model,s));
        assert(st->kda_state[0]==11 && st->kda_window[0]==22 && st->cuda_kda.authority==G53_KDA_DEVICE);
        fail_download_at=0; assert(glm53_kda_ensure_host(&ge->model,s));
        assert(st->kda_state[0]==99 && st->kda_window[0]==199);
    }
    Stream snapshot={.engine=ge}; char error[128]; device_write(s,4,66);
    fail_download_at=downloads+2;
    assert(coli_segment_snapshot(a,write_stream,&snapshot,error,sizeof(error)) != 0 && !snapshot.size);
    float input[4]={0}, output[4]={123,456,123,456};
    ColiSegmentRunRequest run={.struct_size=sizeof(run), .rows=2, .input=input,
        .input_bytes=sizeof(input), .output=output, .output_bytes=sizeof(output)};
    fail_download_at=downloads+1;
    assert(coli_segment_run(a,&run,error,sizeof(error)) != 0);
    assert(!s->filled && !((Glm53SegmentSession *)a->impl)->position && output[0]==123);
    fail_download_at=0;
    assert(!coli_segment_snapshot(a,write_stream,&snapshot,error,sizeof(error)));
    assert(st->kda_state[0]==66 && st->cuda_kda.authority==G53_KDA_BOTH);
    device_write(s,4,88); snapshot.bytes[snapshot.size-1]^=1;
    assert(coli_segment_restore(a,read_stream,&snapshot,error,sizeof(error)) != 0);
    assert(st->cuda_kda.authority==G53_KDA_DEVICE && st->kda_state[0]==66);
    snapshot.bytes[snapshot.size-1]^=1; snapshot.offset=0;
    assert(!coli_segment_restore(a,read_stream,&snapshot,error,sizeof(error)));
    assert(st->cuda_kda.authority==G53_KDA_HOST && st->kda_state[0]==66);
    /* Common pin/capture seam also pulls, and host restore invalidates. */
    Glm53PinState *pin=NULL; device_write(s,4,77);
    assert(glm53_state_capture(&ge->model,&pin,s)); assert(st->kda_state[0]==77);
    device_write(s,4,89); glm53_state_restore(&ge->model,pin,s);
    assert(st->kda_state[0]==77 && st->cuda_kda.authority==G53_KDA_HOST);
    glm53_pin_state_free(pin);
    coli_segment_session_destroy(a); assert(!coli_segment_engine_close(e,NULL,0));
    puts("allocation/pair-transfer failures, atomic pull, snapshot/restore, CPU failure and capture: PASS");
}
static void geometry(void) {
    size_t sb=0,wb=0;
    assert(coli_glm53_cuda_kda_geometry(64,128,8192,4,&sb,&wb));
    assert(sb==4194304 && wb==393216 && sb+wb==4587520);
    assert(!coli_glm53_cuda_kda_geometry(SIZE_MAX,2,0,4,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(1,SIZE_MAX,SIZE_MAX,1,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(SIZE_MAX/2,1,SIZE_MAX/2,1,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(1,1,1,SIZE_MAX,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(1,1,1,SIZE_MAX/3,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(2,3,5,4,&sb,&wb));
    assert(!coli_glm53_cuda_kda_geometry(0,3,0,4,&sb,&wb));
    ColiSegmentEngine *e=engine_open(4,5,4); Glm53SegmentEngine *ge=e->impl;
    ge->model.c.conv_k=1; ColiSegmentSession *s=create(e);
    GLayerState *st=&state(s)->layer[4]; assert(st->cuda_kda.window_bytes==72);
    assert(coli_glm53_cuda_kda_prepare(state(s)->cuda_stage,&st->cuda_kda,st->kda_state,st->kda_window));
    device_write(state(s),4,1); assert(glm53_kda_ensure_host(&ge->model,state(s)));
    coli_segment_session_destroy(s); assert(!coli_segment_engine_close(e,NULL,0));
    /* Partially constructed record is independently safe to close twice. */
    e=engine_open(4,5,4); ge=e->impl;
    ColiGlm53CudaKdaLayer partial={.state=coli_cuda_pipe_alloc(4,16)};
    coli_glm53_cuda_kda_close(&ge->cuda_stage,&partial);
    coli_glm53_cuda_kda_close(&ge->cuda_stage,&partial);
    GSession *host=session_open(&ge->model,8);
    host->cuda_stage=&ge->cuda_stage; host->layer[4].cuda_kda.state=coli_cuda_pipe_alloc(4,16);
    session_close(&ge->model,host);
    assert(!coli_segment_engine_close(e,NULL,0));
    puts("config geometry, overflow, production bytes, K=1 and partial session cleanup: PASS");
}
static void zero_mat(Mat *m, int rows, int columns) {
    m->rows=rows; m->columns=columns; m->f=calloc((size_t)rows*columns,sizeof(float)); assert(m->f);
}
static float *ones(size_t n) {
    float *p=malloc(n*sizeof(float)); assert(p);
    for (size_t i=0; i<n; i++) p[i]=1.f;
    return p;
}
/* Real CPU execution with tiny zero weights: prove pull-before-mutation and
 * host invalidation through the production run callback, without a checkpoint. */
static void cpu_run(void) {
    ColiSegmentEngine *e=engine_open(4,5,4); Glm53SegmentEngine *ge=e->impl;
    Cfg *c=&ge->model.c;
    c->first_dense=32; c->dense_inter=1; c->hc_iters=1;
    c->eps=c->hc_eps=1e-6f; c->gate_lb=-5.f; c->swiglu_limit=1.f;
    GLayer *l=&ge->model.layer[4];
    l->in_ln=ones(2); l->post_ln=ones(2); l->onorm=ones(3);
    l->hc_attn_fn=calloc(6,sizeof(float)); l->hc_ffn_fn=calloc(6,sizeof(float));
    l->hc_attn_base=calloc(3,sizeof(float)); l->hc_ffn_base=calloc(3,sizeof(float));
    l->hc_attn_scale=calloc(3,sizeof(float)); l->hc_ffn_scale=calloc(3,sizeof(float));
    l->dt=calloc(6,sizeof(float)); l->alog=calloc(2,sizeof(float));
    zero_mat(&l->kq,6,2); zero_mat(&l->kk,6,2); zero_mat(&l->kv,6,2); zero_mat(&l->ko,2,6);
    zero_mat(&l->kfa,3,2); zero_mat(&l->kfb,6,3); zero_mat(&l->kga,3,2); zero_mat(&l->kgb,6,3);
    zero_mat(&l->kb,2,2); zero_mat(&l->dg,1,2); zero_mat(&l->du,1,2); zero_mat(&l->dd,2,1);
    ColiSegmentSession *s=create(e); GLayerState *st=&state(s)->layer[4]; device_write(state(s),4,12);
    float input[4]={1,2,2,1},output[4]={0}; char error[128];
    ColiSegmentRunRequest r={.struct_size=sizeof(r), .rows=2, .input=input,
        .input_bytes=sizeof(input), .output=output, .output_bytes=sizeof(output)};
    int u=uploads,d=downloads;
    assert(!coli_segment_run(s,&r,error,sizeof(error)));
    assert(downloads==d+2 && uploads==u && state(s)->filled==2);
    assert(st->cuda_kda.authority==G53_KDA_HOST && st->kda_state[0]>0 && st->kda_state[0]<12);
    assert(st->kda_window[3]==0 && ((float *)st->cuda_kda.state)[0]==12);
    assert(isfinite(output[0]) && isfinite(output[1]));
    assert(coli_glm53_cuda_kda_prepare(state(s)->cuda_stage,&st->cuda_kda,st->kda_state,st->kda_window));
    assert(uploads==u+2 && !memcmp(st->cuda_kda.state,st->kda_state,72));
    coli_segment_session_destroy(s); assert(!coli_segment_engine_close(e,NULL,0));
    puts("real CPU KDA run: DEVICE pull before execution, host mutation marks HOST, reprepare exact: PASS");
}
int main(void) {
    assert(!setenv("COLI_CUDA","1",1)); assert(!setenv("COLI_GPUS","2,4,6",1)); unsetenv("COLI_GPU");
    assert(!coli_segment_adapter_register(&adapter));
    geometry(); range_check(4,8,4); range_check(7,13,4); range_check(20,28,4);
    range_check(6,7,4); /* full-only */
    int n=allocs,u=uploads,d=downloads,a=acquires,f=frees;
    range_check(4,8,-1);
    assert(allocs==n && uploads==u && downloads==d && acquires==a && frees==f);
    failures(); cpu_run();
    ColiSegmentEngine *ea=engine_open(4,5,2), *eb=engine_open(7,8,6);
    ColiSegmentSession *sa=create(ea), *sb=create(eb);
    coli_segment_session_destroy(sa); assert(!coli_segment_engine_close(ea,NULL,0));
    assert(lifetime.users==1); device_write(state(sb),7,123);
    assert(glm53_kda_ensure_host(&((Glm53SegmentEngine *)eb->impl)->model,state(sb)));
    coli_segment_session_destroy(sb); assert(!coli_segment_engine_close(eb,NULL,0));
    assert(!live && !lifetime.users && inits==shutdowns);
    assert(nevents>=2 && events[nevents-2].op=='R' && events[nevents-1].op=='S');
    ea=engine_open(4,5,4); sa=create(ea); coli_segment_session_destroy(sa);
    assert(!coli_segment_engine_close(ea,NULL,0)); assert(inits==shutdowns && !live);
    puts("shared parent lease, different owners, survivor, final release and reacquire: PASS");
    puts("GLM53 CUDA KDA session state/coherence: PASS");
    return 0;
}
