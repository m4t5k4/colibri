/* Real Segment callbacks + shared production lifetime/coherence, with the
 * mirror test's allocation/transfer fake extended by its scalar recurrence. */
#define main mirror_probe_main
#include "test_glm53_cuda_kda_state.c"
#undef main

static void weights(Mat *m,int rows,int columns,int salt) {
    zero_mat(m,rows,columns);
    for (int i=0; i<rows*columns; i++) ((float *)m->f)[i]=(float)((i*7+salt)%17-8)*0.045f;
}
static ColiSegmentEngine *numerical_engine(int owner,int layers) {
    ColiSegmentEngine *e=engine_open(4,4+layers,owner);
    Glm53SegmentEngine *ge=e->impl; Cfg *c=&ge->model.c;
    c->first_dense=32; c->dense_inter=3; c->hc_iters=2;
    c->eps=0.003f; c->hc_eps=1e-6f; c->gate_lb=-5.f; c->swiglu_limit=4.f;
    ge->model.layer=calloc(32,sizeof(GLayer)); assert(ge->model.layer);
    for (int i=4; i<4+layers; i++) {
        GLayer *l=&ge->model.layer[i];
        l->in_ln=ones(2); l->post_ln=ones(2); l->onorm=ones(3);
        l->hc_attn_fn=calloc(6,sizeof(float)); l->hc_ffn_fn=calloc(6,sizeof(float));
        l->hc_attn_base=calloc(3,sizeof(float)); l->hc_ffn_base=calloc(3,sizeof(float));
        l->hc_attn_scale=calloc(3,sizeof(float)); l->hc_ffn_scale=calloc(3,sizeof(float));
        l->conv=calloc(72,sizeof(float)); l->dt=calloc(6,sizeof(float)); l->alog=calloc(2,sizeof(float));
        assert(l->conv && l->dt && l->alog);
        for (int j=0; j<72; j++) ((float *)l->conv)[j]=(float)((j*3+i)%11-5)*0.09f;
        for (int j=0; j<6; j++) ((float *)l->dt)[j]=(float)(j-2)*0.13f;
        ((float *)l->alog)[0]=-0.25f; ((float *)l->alog)[1]=0.3f;
        weights(&l->kq,6,2,i); weights(&l->kk,6,2,i+2); weights(&l->kv,6,2,i+4);
        weights(&l->ko,2,6,i+6); weights(&l->kfa,3,2,i+8); weights(&l->kfb,6,3,i+10);
        weights(&l->kga,3,2,i+1); weights(&l->kgb,6,3,i+3); weights(&l->kb,2,2,i+5);
        weights(&l->dg,3,2,i+7); weights(&l->du,3,2,i+9); weights(&l->dd,2,3,i+11);
    }
    return e;
}
static void seed(ColiSegmentSession *s,int salt) {
    GSession *gs=state(s); Glm53SegmentEngine *ge=((Glm53SegmentSession *)s->impl)->engine;
    for (int i=ge->layer_begin; i<(int)ge->layer_end; i++) {
        GLayerState *l=&gs->layer[i];
        for (int j=0; j<18; j++) l->kda_state[j]=(float)((j+salt)%13-6)*0.015f;
        for (int j=0; j<72; j++) l->kda_window[j]=(float)((j*3+salt)%19-9)*0.023f;
    }
}
static int run(ColiSegmentSession *s,int rows,int salt,float *out,char *error) {
    float input[16];
    for (int j=0; j<2*rows; j++) input[j]=(float)((j*5+salt)%17-8)*0.11f;
    ColiSegmentRunRequest r={.struct_size=sizeof(r),.rows=(uint32_t)rows,
        .position=((Glm53SegmentSession *)s->impl)->position,
        .input=input,.input_bytes=(size_t)rows*2*sizeof(float),
        .output=out,.output_bytes=(size_t)rows*2*sizeof(float)};
    watch_session=state(s)->cuda_stage?state(s):NULL;
    int result=coli_segment_run(s,&r,error,128); watch_session=NULL; watch_layer=NULL;
    return result;
}
static void close_engine(ColiSegmentEngine *e) { assert(!coli_segment_engine_close(e,NULL,0)); }
static void near(const float *a,const float *b,size_t n) {
    for (size_t i=0; i<n; i++) assert(isfinite(a[i]) && isfinite(b[i]) &&
        fabsf(a[i]-b[i])<=1e-5f*(1.f+fabsf(b[i])));
}
static void equal_session(ColiSegmentSession *gpu,ColiSegmentSession *cpu) {
    GSession *a=state(gpu),*b=state(cpu);
    Glm53SegmentSession *ga=gpu->impl,*cb=cpu->impl;
    assert(ga->position==cb->position && a->filled==b->filled);
    for (int i=ga->engine->layer_begin; i<(int)ga->engine->layer_end; i++) {
        GLayerState *l=&a->layer[i],*r=&b->layer[i];
        const float *s=l->cuda_kda.authority==G53_KDA_DEVICE ? l->cuda_kda.state : l->kda_state;
        near(s,r->kda_state,18); assert(!memcmp(l->kda_window,r->kda_window,288));
        if (l->cuda_kda.authority==G53_KDA_DEVICE || l->cuda_kda.authority==G53_KDA_BOTH)
            assert(!memcmp(l->cuda_kda.window,r->kda_window,288));
    }
}
static void matched_run(ColiSegmentSession *gpu,ColiSegmentSession *cpu,int rows,int salt) {
    float a[16]={0},b[16]={0}; char error[128]; int n=allocs,f=frees;
    float *device=state(gpu)->kda_staging.device,*tmp=state(gpu)->kda_staging.next_window;
    assert(!run(cpu,rows,salt,b,error));
    assert(!run(gpu,rows,salt,a,error)); near(a,b,(size_t)rows*2); equal_session(gpu,cpu);
    assert(allocs==n && frees==f && state(gpu)->kda_staging.device==device && state(gpu)->kda_staging.next_window==tmp);
}
static void capture(ColiSegmentSession *s,Stream *stream) {
    stream->engine=((Glm53SegmentSession *)s->impl)->engine; stream->size=stream->offset=0;
    char error[128]; assert(!coli_segment_snapshot(s,write_stream,stream,error,sizeof(error)));
}
static int restore(ColiSegmentSession *s,Stream *stream) {
    stream->engine=((Glm53SegmentSession *)s->impl)->engine; stream->offset=0;
    char error[128];return coli_segment_restore(s,read_stream,stream,error,sizeof(error));
}
static void shortconv_equivalence(void) {
    for (int production=0; production<2; production++) for (int kernel=1; kernel<=4; kernel+=3) {
        int h=production?64:2,d=production?128:3,p=h*d;
        size_t nw=(size_t)3*p*kernel,ns=(size_t)p*d;
        float *w=malloc(nw*4),*ref=malloc(nw*4),*conv=malloc(nw*4),*qkv=malloc((size_t)3*p*4);
        float *mixed=malloc((size_t)3*p*4),*scratch=malloc((size_t)coli_kda_scratch_floats(h,d,d)*4);
        float *st=calloc(ns,4),*out=malloc((size_t)p*4),*decay=calloc((size_t)p,4),*beta=calloc((size_t)h,4);
        assert(w && ref && conv && qkv && mixed && scratch && st && out && decay && beta);
        for (size_t j=0; j<nw; j++) { w[j]=(float)((int)(j%17)-8)*0.027f; conv[j]=(float)((int)(j%11)-5)*0.06f; }
        for (int j=0; j<3*p; j++) qkv[j]=(float)(j%23-11)*0.037f;
        memcpy(ref,w,nw*4);
        glm53_kda_shortconv(mixed,w,qkv,conv,p,kernel);
        assert(!coli_kda_step(out,st,ref,qkv,conv,decay,beta,h,d,d,kernel,1e-6f,scratch));
        assert(!memcmp(w,ref,nw*4) && !memcmp(mixed,scratch,(size_t)3*p*4));
        printf("ShortConv H=%d D=%d K=%d: exact window/mixed bytes PASS\n",h,d,kernel);
        free(w);free(ref);free(conv);free(qkv);free(mixed);free(scratch);free(st);free(out);free(decay);free(beta);
    }
    size_t db,wb;assert(coli_glm53_cuda_kda_staging_geometry(64,128,8192,4,&db,&wb));
    assert(db==164096 && wb==393216);
    assert(!coli_glm53_cuda_kda_staging_geometry(2,3,5,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(1,257,257,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(65536,1,65536,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(SIZE_MAX,2,0,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(64,128,8192,SIZE_MAX,&db,&wb));
}
static void sequences(void) {
    ColiSegmentEngine *e=numerical_engine(4,2),*cpu=numerical_engine(-1,2);
    ColiSegmentSession *a=create(e),*b=create(e),*ca=create(cpu),*cb=create(cpu);
    seed(a,1);seed(ca,1);seed(b,4);seed(cb,4);
    assert(state(a)->kda_staging.device!=state(b)->kda_staging.device);
    assert(state(a)->kda_staging.next_window!=state(b)->kda_staging.next_window);
    assert(state(a)->layer[4].cuda_kda.state!=state(b)->layer[4].cuda_kda.state);
    int u=uploads,d=downloads,k=recurrence_calls,y=sync_calls;
    matched_run(a,ca,1,1);
    assert(uploads==u+16 && downloads==d+2 && recurrence_calls==k+2 && sync_calls==y+2);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_DEVICE);
    matched_run(b,cb,1,5);
    u=uploads;d=downloads; matched_run(a,ca,1,3);
    assert(uploads==u+12 && downloads==d+2); /* no recurrence/window D2H on next decode */
    Stream sa={0},sb={0}; d=downloads;capture(b,&sb);capture(a,&sa);assert(downloads==d+8);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_BOTH);
    equal_session(a,ca);equal_session(b,cb);
    /* Exercise DEVICE -> BOTH -> HOST without snapshot intervening. */
    matched_run(a,ca,1,7);d=downloads;u=uploads;k=recurrence_calls;
    matched_run(a,ca,2,9);assert(downloads==d+4 && uploads==u && recurrence_calls==k);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    u=uploads;d=downloads;matched_run(a,ca,1,11);assert(uploads==u+16 && downloads==d+2);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_DEVICE);
    /* Restore prior snapshot into both sequences, then continue numerically. */
    assert(!restore(a,&sa));assert(!restore(ca,&sa));equal_session(a,ca);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    matched_run(a,ca,1,13);capture(a,&sa);
    /* Complete overwrite from every valid authority. */
    for (int auth=G53_KDA_HOST; auth<=G53_KDA_BOTH; auth++) {
        state(a)->layer[4].cuda_kda.authority=(ColiGlm53KdaAuthority)auth;
        assert(!restore(a,&sa));assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    }
    printf("2-layer Segment: fresh decode U=16/D=2/K=2/sync=2; resident decode U=12/D=2; CPU prefill D=4/U=0; snapshot D=4: PASS\n");
    puts("CPU/GPU sequences, prefill both directions, snapshot/restore continuation, interleaved isolated sessions: PASS");
    coli_segment_session_destroy(a);assert(e->active_sessions==1 && lifetime.users==1);
    matched_run(b,cb,1,15);coli_segment_session_destroy(b);close_engine(e);
    coli_segment_session_destroy(ca);coli_segment_session_destroy(cb);close_engine(cpu);
}
static void refusals(ColiSegmentSession *s,Stream *trusted) {
    GSession *gs=state(s);GLayerState *l=&gs->layer[4]; char error[128];
    assert(l->cuda_kda.authority==G53_KDA_UNKNOWN);
    int u=uploads,d=downloads,k=recurrence_calls;float out[4]={91,92,93,94};
    assert(!coli_glm53_cuda_kda_prepare(gs->cuda_stage,&l->cuda_kda,l->kda_state,l->kda_window));
    assert(!coli_glm53_cuda_kda_ensure_host(gs->cuda_stage,&l->cuda_kda,l->kda_state,l->kda_window));
    assert(run(s,1,7,out,error) && strstr(error,"indeterminate"));
    assert(run(s,2,7,out,error) && out[0]==91 && out[3]==94);
    Stream sink={.engine=((Glm53SegmentSession *)s->impl)->engine};
    assert(coli_segment_snapshot(s,write_stream,&sink,error,sizeof(error)) && !sink.size);
    assert(uploads==u && downloads==d && recurrence_calls==k);
    float saved_s[18],saved_w[72];memcpy(saved_s,l->kda_state,72);memcpy(saved_w,l->kda_window,288);
    trusted->bytes[trusted->size-1]^=1;
    assert(restore(s,trusted));assert(l->cuda_kda.authority==G53_KDA_UNKNOWN);
    assert(!memcmp(saved_s,l->kda_state,72) && !memcmp(saved_w,l->kda_window,288));
    trusted->bytes[trusted->size-1]^=1;
    size_t full=trusted->size;trusted->size--;assert(restore(s,trusted));trusted->size=full;
    assert(l->cuda_kda.authority==G53_KDA_UNKNOWN && !memcmp(saved_w,l->kda_window,288));
    assert(!restore(s,trusted));assert(l->cuda_kda.authority==G53_KDA_HOST);
}
static void fail_case(int authority,int seam) {
    ColiSegmentEngine *e=numerical_engine(4,1),*cpu=numerical_engine(-1,1);
    ColiSegmentSession *s=create(e),*c=create(cpu);seed(s,2);seed(c,2);
    GSession *gs=state(s);GLayerState *l=&gs->layer[4];
    if (authority==G53_KDA_DEVICE) matched_run(s,c,1,2);
    Stream trusted={0};capture(s,&trusted); /* trusted complete pair for recovery */
    if (authority==G53_KDA_HOST) l->cuda_kda.authority=G53_KDA_HOST;
    else {
        assert(coli_glm53_cuda_kda_prepare(gs->cuda_stage,&l->cuda_kda,l->kda_state,l->kda_window));
        l->cuda_kda.authority=(ColiGlm53KdaAuthority)authority;
    }
    float saved_s[18],saved_w[72];memcpy(saved_s,l->kda_state,72);memcpy(saved_w,l->kda_window,288);
    uint32_t pos=((Glm53SegmentSession *)s->impl)->position;int filled=gs->filled;
    /* 0/1 = prepare state/window; 2..6 = staged inputs; 7 = reject launch;
     * 8 = sync; 9 = raw D2H; 10 = next-window H2D after accepted mutation. */
    int prepare=authority==G53_KDA_HOST?2:0;
    if (seam<2) { assert(prepare);fail_upload_at=uploads+seam+1; }
    else if (seam<=6) fail_upload_at=uploads+prepare+seam-1;
    else if (seam==7) fail_recurrence=1;
    else if (seam==8) fail_sync=1;
    else if (seam==9) fail_download_at=downloads+1;
    else fail_upload_at=uploads+prepare+6;
    float out[2]={91,92};char error[128];assert(run(s,1,5,out,error));
    fail_upload_at=fail_download_at=fail_recurrence=fail_sync=0;
    assert(!memcmp(saved_s,l->kda_state,72) && !memcmp(saved_w,l->kda_window,288));
    assert(pos==((Glm53SegmentSession *)s->impl)->position && filled==gs->filled && out[0]==91 && out[1]==92);
    if (seam>=8) {
        assert(strstr(error,"indeterminate"));refusals(s,&trusted);
    } else {
        assert(strstr(error,"before mutation") && l->cuda_kda.authority==(ColiGlm53KdaAuthority)authority);
        assert(!restore(s,&trusted));
    }
    /* Complete restore is a usable recovery, not just an enum change. */
    assert(!restore(c,&trusted));matched_run(s,c,1,7);
    coli_segment_session_destroy(s);coli_segment_session_destroy(c);close_engine(e);close_engine(cpu);
}
/* A failed range is post-mutation once any earlier layer has committed.
 * The still-unlaunched layer keeps valid authority; completed layers are fenced. */
static void later_layer_failure(void) {
    for (int post=0; post<2; post++) {
        ColiSegmentEngine *e=numerical_engine(4,2),*cpu=numerical_engine(-1,2);
        ColiSegmentSession *s=create(e),*c=create(cpu);seed(s,2);seed(c,2);
        Stream trusted={0};capture(s,&trusted);
        /* First fresh layer uses 8 uploads. Second fails q upload (11th) or
         * window publication (16th, after accepted mutation). */
        fail_upload_at=uploads+(post?16:11);
        float out[2]={91,92};char error[128];assert(run(s,1,3,out,error));fail_upload_at=0;
        GSession *gs=state(s);
        assert(!gs->filled && !((Glm53SegmentSession *)s->impl)->position && out[0]==91);
        assert(gs->layer[4].cuda_kda.authority==G53_KDA_UNKNOWN);
        assert(gs->layer[5].cuda_kda.authority==(post?G53_KDA_UNKNOWN:G53_KDA_HOST));
        int k=recurrence_calls;
        assert(run(s,1,5,out,error) && recurrence_calls==k);
        Stream sink={.engine=e->impl};
        assert(coli_segment_snapshot(s,write_stream,&sink,error,sizeof(error)) && !sink.size);
        assert(!restore(s,&trusted));assert(!restore(c,&trusted));matched_run(s,c,1,7);
        coli_segment_session_destroy(s);coli_segment_session_destroy(c);close_engine(e);close_engine(cpu);
    }
    puts("Later-layer pre/post-launch failure: earlier accepted updates UNKNOWN; no partial-prefix snapshot/replay; restore continuation PASS");
}
/* A CPU prefill may mutate an earlier layer before a later D2H fails. */
static void later_prefill_failure(void) {
    ColiSegmentEngine *e=numerical_engine(4,2),*cpu=numerical_engine(-1,2);
    ColiSegmentSession *s=create(e),*c=create(cpu);seed(s,2);seed(c,2);
    matched_run(s,c,1,2);
    Stream trusted={0};capture(s,&trusted);
    matched_run(s,c,1,3); /* both layers DEVICE */
    uint32_t pos=((Glm53SegmentSession *)s->impl)->position;int filled=state(s)->filled;
    fail_download_at=downloads+3; /* layer A pulled and mutated; B pull fails */
    float out[4]={91,92,93,94};char error[128];assert(run(s,2,4,out,error));fail_download_at=0;
    assert(pos==((Glm53SegmentSession *)s->impl)->position && filled==state(s)->filled);
    assert(run(s,1,5,out,error)); /* mixed token positions must be fenced */
    Stream sink={.engine=e->impl};
    assert(coli_segment_snapshot(s,write_stream,&sink,error,sizeof(error)) && !sink.size);
    assert(!restore(s,&trusted));assert(!restore(c,&trusted));matched_run(s,c,1,7);
    coli_segment_session_destroy(s);coli_segment_session_destroy(c);close_engine(e);close_engine(cpu);
    puts("Later CPU-prefill layer pull failure: session fenced; complete restore continuation PASS");
}
static void pins_and_late_unknown(void) {
    ColiSegmentEngine *e=numerical_engine(4,2);ColiSegmentSession *s=create(e);seed(s,3);
    GSession *gs=state(s);GModel *m=&((Glm53SegmentEngine *)e->impl)->model;Glm53PinState *pin=NULL;
    assert(glm53_state_capture(m,&pin,gs));gs->layer[5].cuda_kda.authority=G53_KDA_UNKNOWN;
    float out[2];char error[128];int calls=recurrence_calls;
    assert(run(s,1,4,out,error) && recurrence_calls==calls); /* earlier layer cannot mutate */
    float *saved=pin->window[5];pin->window[5]=NULL;
    glm53_state_restore(m,pin,gs);assert(gs->layer[5].cuda_kda.authority==G53_KDA_UNKNOWN);
    pin->window[5]=saved;glm53_state_restore(m,pin,gs);
    assert(gs->layer[4].cuda_kda.authority==G53_KDA_HOST && gs->layer[5].cuda_kda.authority==G53_KDA_HOST);
    glm53_pin_state_free(pin);coli_segment_session_destroy(s);
    s=create(e);assert(state(s)->layer[4].cuda_kda.authority==G53_KDA_HOST);coli_segment_session_destroy(s);close_engine(e);
    puts("UNKNOWN blocks earlier layers; malformed pin cannot recover; complete pin/reset can recover: PASS");
}
int main(void) {
    setenv("COLI_CUDA","1",1);setenv("COLI_GPUS","2,4,6",1);unsetenv("COLI_GPU");
    assert(!coli_segment_adapter_register(&adapter));
    shortconv_equivalence();sequences();
    for (int authority=G53_KDA_HOST; authority<=G53_KDA_BOTH; authority++)
        for (int seam=authority==G53_KDA_HOST?0:2; seam<=10; seam++) fail_case(authority,seam);
    puts("29 transaction failure cases: canonical host bytes/position unchanged, authority/recovery correct: PASS");
    later_layer_failure(); later_prefill_failure(); pins_and_late_unknown();
    assert(!live && !lifetime.users && inits==shutdowns);
    printf("Fake totals allocations=%d frees=%d uploads=%d downloads=%d recurrence=%d sync=%d; owner 4 in {2,4,6}; no extra lease: PASS\n",
        allocs,frees,uploads,downloads,recurrence_calls,sync_calls);
    puts("GLM53 CUDA recurrence production Segment integration: PASS");return 0;
}
