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
    synthetic_numeric=1;
    ColiSegmentEngine *e=engine_open(4,4+layers,owner);synthetic_numeric=0;
    Glm53SegmentEngine *ge=e->impl; Cfg *c=&ge->model.c;
    c->first_dense=32; c->dense_inter=3; c->hc_iters=2;
    c->eps=0.003f; c->hc_eps=1e-6f; c->gate_lb=-5.f; c->swiglu_limit=4.f;
    assert(ge->model.layer);
    for (int i=4; i<4+layers; i++) {
        GLayer *l=&ge->model.layer[i];
        l->in_ln=ones(2); l->post_ln=ones(2); l->onorm=ones(3);
        l->hc_attn_fn=calloc(6,sizeof(float)); l->hc_ffn_fn=calloc(6,sizeof(float));
        l->hc_attn_base=calloc(3,sizeof(float)); l->hc_ffn_base=calloc(3,sizeof(float));
        l->hc_attn_scale=calloc(3,sizeof(float)); l->hc_ffn_scale=calloc(3,sizeof(float));
        l->dt=calloc(6,sizeof(float)); l->alog=calloc(2,sizeof(float));
        assert(l->conv && l->dt && l->alog);
        /* Host/device convolution was initialized once by engine setup. */
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
    watch_engine=watch_session?((Glm53SegmentSession *)s->impl)->engine:NULL;
    int result=coli_segment_run(s,&r,error,128);watch_session=NULL;watch_engine=NULL;watch_layer=NULL;
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
        const float *window=l->cuda_kda.authority==G53_KDA_DEVICE ? l->cuda_kda.window : l->kda_window;
        near(s,r->kda_state,18);assert(!memcmp(window,r->kda_window,288));
        if (l->cuda_kda.authority==G53_KDA_DEVICE || l->cuda_kda.authority==G53_KDA_BOTH)
            assert(!memcmp(l->cuda_kda.window,r->kda_window,288));
    }
}
static void matched_run(ColiSegmentSession *gpu,ColiSegmentSession *cpu,int rows,int salt) {
    float a[16]={0},b[16]={0}; char error[128]; int n=allocs,f=frees;
    float *device=state(gpu)->kda_staging.device;
    assert(!run(cpu,rows,salt,b,error));
    assert(!run(gpu,rows,salt,a,error)); near(a,b,(size_t)rows*2); equal_session(gpu,cpu);
    assert(allocs==n && frees==f && state(gpu)->kda_staging.device==device);
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
    assert(db==262400 && wb==393216);
    size_t sb,pair;
    assert(coli_glm53_cuda_kda_geometry(64,128,8192,4,&sb,&pair));
    assert(sb==4194304 && pair==393216 && 34*pair==13369344);
    assert(!coli_glm53_cuda_kda_staging_geometry(2,3,5,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(1,257,257,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(65536,1,65536,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(SIZE_MAX,2,0,4,&db,&wb));
    assert(!coli_glm53_cuda_kda_staging_geometry(64,128,8192,SIZE_MAX,&db,&wb));
}
static void counted_token(ColiSegmentSession *s,ColiSegmentSession *control,int salt,int fresh) {
    GSession *gs=state(s);Glm53SegmentEngine *ge=((Glm53SegmentSession *)s->impl)->engine;
    int count=(int)ge->layer_end-(int)ge->layer_begin;
    int u=uploads,d=downloads,c=shortconv_calls,k=recurrence_calls,y=sync_calls;
    size_t ub=upload_bytes,db=download_bytes,start=nevents;
    matched_run(s,control,1,salt);
    assert(uploads==u+count*(fresh?5:3) && downloads==d+count);
    assert(shortconv_calls==c+count && recurrence_calls==k+count && sync_calls==y+count);
    assert(upload_bytes-ub==(size_t)count*(104+(fresh?360:0)) && download_bytes-db==(size_t)count*24);
    size_t at=start;
    for (int i=ge->layer_begin;i<(int)ge->layer_end;i++) {
        ColiGlm53CudaKdaLayer *l=&gs->layer[i].cuda_kda;
        if (fresh) {
            assert(events[at].op=='U' && events[at].p==l->state && events[at++].bytes==72);
            assert(events[at].op=='U' && events[at].p==l->window && events[at++].bytes==288);
        }
        assert(events[at].op=='U' && events[at].p==gs->kda_staging.projected_qkv && events[at++].bytes==72);
        assert(events[at].op=='U' && events[at].p==gs->kda_staging.decay && events[at++].bytes==24);
        assert(events[at].op=='U' && events[at].p==gs->kda_staging.beta && events[at++].bytes==8);
        assert(events[at].op=='C' && events[at++].p==l->window);
        assert(events[at].op=='K' && events[at++].p==l->state);
        assert(events[at++].op=='Y');
        assert(events[at].op=='D' && events[at].p==gs->kda_staging.raw_out && events[at++].bytes==24);
        assert(l->authority==G53_KDA_DEVICE);
    }
    assert(at==nevents); /* no hidden pair/conv transfers, allocations or intermediate sync */
}
static void sequences(void) {
    int weights_before=uploads;size_t weights_bytes=upload_bytes;
    ColiSegmentEngine *e=numerical_engine(4,2),*cpu=numerical_engine(-1,2);
    Glm53SegmentEngine *ge=e->impl;
    assert(uploads==weights_before+2 && upload_bytes-weights_bytes==576);
    void *conv0=ge->model.layer[4].cuda_kda_weights.conv,*conv1=ge->model.layer[5].cuda_kda_weights.conv;
    assert(conv0!=conv1 && find(4,conv0,288) && find(4,conv1,288));
    int setup_uploads=uploads;
    ColiSegmentSession *a=create(e),*b=create(e),*ca=create(cpu),*cb=create(cpu);
    assert(uploads==setup_uploads); /* sessions borrow the two immutable copies */
    seed(a,1);seed(ca,1);seed(b,4);seed(cb,4);
    assert(state(a)->kda_staging.device!=state(b)->kda_staging.device);
    assert(state(a)->layer[4].cuda_kda.state!=state(b)->layer[4].cuda_kda.state);
    assert(state(a)->layer[4].cuda_kda.window!=state(b)->layer[4].cuda_kda.window);
    for (int i=4;i<6;i++) assert(coli_glm53_cuda_kda_prepare(state(a)->cuda_stage,
        &state(a)->layer[i].cuda_kda,state(a)->layer[i].kda_state,state(a)->layer[i].kda_window));
    float old_state[2][18],old_window[2][72];
    for (int i=4;i<6;i++) {memcpy(old_state[i-4],state(a)->layer[i].kda_state,72);memcpy(old_window[i-4],state(a)->layer[i].kda_window,288);}
    counted_token(a,ca,1,0);counted_token(b,cb,5,1);counted_token(a,ca,3,0);counted_token(a,ca,7,0);
    for (int i=4;i<6;i++) {
        assert(!memcmp(old_state[i-4],state(a)->layer[i].kda_state,72));
        assert(!memcmp(old_window[i-4],state(a)->layer[i].kda_window,288));
        assert(memcmp(old_window[i-4],state(a)->layer[i].cuda_kda.window,288));
        assert(memcmp(old_state[i-4],state(a)->layer[i].cuda_kda.state,72));
    }
    Stream sa={0},sb={0};int d=downloads;capture(b,&sb);capture(a,&sa);assert(downloads==d+8);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_BOTH);
    equal_session(a,ca);equal_session(b,cb);
    assert(!memcmp(state(a)->layer[4].kda_state,state(a)->layer[4].cuda_kda.state,72));
    assert(!memcmp(state(a)->layer[4].kda_window,state(a)->layer[4].cuda_kda.window,288));
    counted_token(b,cb,9,0);counted_token(a,ca,7,0);
    d=downloads;int u=uploads,k=recurrence_calls,c=shortconv_calls;
    matched_run(a,ca,2,9);assert(downloads==d+4 && uploads==u && recurrence_calls==k && shortconv_calls==c);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    counted_token(a,ca,11,1);
    assert(!restore(a,&sa));assert(!restore(ca,&sa));equal_session(a,ca);
    assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    counted_token(a,ca,13,1);capture(a,&sa);
    for (int auth=G53_KDA_HOST;auth<=G53_KDA_BOTH;auth++) {
        state(a)->layer[4].cuda_kda.authority=(ColiGlm53KdaAuthority)auth;
        assert(!restore(a,&sa));assert(state(a)->layer[4].cuda_kda.authority==G53_KDA_HOST);
    }
    puts("Two KDA layers: each resident token U=6/208 bytes, D=2/48 bytes, ShortConv=2/recur=2/sync=2; three consecutive tokens: PASS");
    puts("BOTH -> DEVICE leaves both host components stale; complete snapshot restores BOTH: PASS");
    puts("CPU/GPU transitions, restore continuation and shared-conv/isolated-session interleave: PASS");
    coli_segment_session_destroy(a);assert(e->active_sessions==1 && lifetime.users==1);
    assert(ge->model.layer[4].cuda_kda_weights.conv==conv0 && find(4,conv0,288));
    counted_token(b,cb,15,0);coli_segment_session_destroy(b);
    assert(find(4,conv0,288) && find(4,conv1,288));close_engine(e);
    coli_segment_session_destroy(ca);coli_segment_session_destroy(cb);close_engine(cpu);
}
static int pointer_live(const void *p) {
    for (int i=0;i<128;i++) if (buffers[i].p==p) return 1;
    return 0;
}
static void static_lifetime(void) {
    for (int reverse=0;reverse<2;reverse++) {
        int u=uploads;size_t ub=upload_bytes,first=nevents;
        ColiSegmentEngine *e=numerical_engine(4,2);Glm53SegmentEngine *ge=e->impl;
        void *p0=ge->model.layer[4].cuda_kda_weights.conv,*p1=ge->model.layer[5].cuda_kda_weights.conv;
        assert(uploads==u+2 && upload_bytes-ub==576);
        assert(ge->model.layer[4].cuda_kda_weights.bytes==288 && ge->model.layer[5].cuda_kda_weights.bytes==288);
        assert(find(4,p0,288) && find(4,p1,288));
        assert(!memcmp(p0,ge->model.layer[4].conv,288) && !memcmp(p1,ge->model.layer[5].conv,288));
        ColiSegmentSession *a=create(e),*b=create(e);assert(uploads==u+2);
        assert(state(a)->layer[4].cuda_kda.state!=state(b)->layer[4].cuda_kda.state);
        assert(state(a)->layer[4].cuda_kda.window!=state(b)->layer[4].cuda_kda.window);
        assert(state(a)->kda_staging.device!=state(b)->kda_staging.device);
        assert(coli_segment_engine_close(e,NULL,0));
        coli_segment_session_destroy(reverse?b:a);assert(pointer_live(p0) && pointer_live(p1));
        coli_segment_session_destroy(reverse?a:b);assert(pointer_live(p0) && pointer_live(p1));
        assert(live==3); /* stage wire and two static convolution copies */
        int seen0=0,seen1=0;
        for (size_t i=first;i<nevents;i++) if (events[i].op=='U') {
            seen0+=events[i].p==p0;seen1+=events[i].p==p1;
        }
        assert(seen0==1 && seen1==1);
        size_t closing=nevents;close_engine(e);
        assert(!pointer_live(p0) && !pointer_live(p1));
        int freed0=0,freed1=0;size_t release=0;
        for (size_t i=closing;i<nevents;i++) {
            if (events[i].op=='F') {freed0+=events[i].p==p0;freed1+=events[i].p==p1;}
            if (events[i].op=='R') release=i;
        }
        assert(freed0==1 && freed1==1 && release>closing && events[release+1].op=='S');
    }
    puts("Static conv: 576 bytes/2-layer fixture; upload once; shared by sessions; both close orders; model free before final lease: PASS");
}
static void static_setup_failure(void) {
    ColiSegmentEngine *survivor=numerical_engine(4,1);ColiSegmentSession *s=create(survivor);
    int baseline=live;
    for (int mode=0;mode<2;mode++) for (int layer=1;layer<=2;layer++) {
        ColiGlm53StagePlan plan={sizeof(plan),COLI_GLM53_STAGE_PLAN_VERSION,4};
        ColiSegmentEngineOptions opts={.struct_size=sizeof(opts),.model_dir="synthetic",
            .layer_begin=4,.layer_end=6,.context_tokens=8,.resource_plan=&plan,.resource_plan_size=sizeof(plan)};
        if (mode==0) fail_alloc_at=allocs+1+layer; /* stage wire then static conv */
        else fail_upload_at=uploads+layer;
        ColiSegmentEngine *bad=NULL;char error[128];synthetic_numeric=1;
        assert(coli_segment_engine_open("kda-test",&opts,&bad,error,sizeof(error)) && !bad);
        synthetic_numeric=0;fail_alloc_at=fail_upload_at=0;
        assert(live==baseline && lifetime.users==1 && survivor->active_sessions==1);
    }
    float output[2];char error[128];seed(s,2);assert(!run(s,1,3,output,error));
    coli_segment_session_destroy(s);close_engine(survivor);
    puts("Static conv partial allocation/upload failures: engine fails cleanly; existing session/lease survives: PASS");
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
    /* 0/1 = prepare state/window; 2..4 = qkv/decay/beta; 5 = SC rejection;
     * 6 = recurrence rejection AFTER SC acceptance; 7 = sync; 8 = raw D2H. */
    int prepare=authority==G53_KDA_HOST?2:0;
    if (seam<2) { assert(prepare);fail_upload_at=uploads+seam+1; }
    else if (seam<=4) fail_upload_at=uploads+prepare+seam-1;
    else if (seam==5) fail_shortconv=1;
    else if (seam==6) fail_recurrence=1;
    else if (seam==7) fail_sync=1;
    else fail_download_at=downloads+1;
    float out[2]={91,92};char error[128];assert(run(s,1,5,out,error));
    fail_upload_at=fail_download_at=fail_recurrence=fail_shortconv=fail_sync=0;
    assert(!memcmp(saved_s,l->kda_state,72) && !memcmp(saved_w,l->kda_window,288));
    assert(pos==((Glm53SegmentSession *)s->impl)->position && filled==gs->filled && out[0]==91 && out[1]==92);
    if (seam>=6) {
        assert(gs->kda_poisoned);
        assert(strstr(error,"indeterminate"));refusals(s,&trusted);
    } else {
        assert(!gs->kda_poisoned);
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
        /* Layer A commits; B rejects qkv staging or rejects recurrence after SC. */
        if (post) fail_recurrence_at=recurrence_calls+2;
        else fail_upload_at=uploads+8; /* five A uploads, two B prepare uploads, then B qkv */
        float out[2]={91,92};char error[128];assert(run(s,1,3,out,error));
        fail_upload_at=fail_recurrence_at=0;
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
    shortconv_equivalence();static_lifetime();static_setup_failure();sequences();
    for (int authority=G53_KDA_HOST; authority<=G53_KDA_BOTH; authority++)
        for (int seam=authority==G53_KDA_HOST?0:2; seam<=8; seam++) fail_case(authority,seam);
    puts("23 transaction failure cases: canonical host bytes/position unchanged, authority/recovery correct: PASS");
    later_layer_failure(); later_prefill_failure(); pins_and_late_unknown();
    assert(!live && !lifetime.users && inits==shutdowns);
    printf("Fake totals alloc attempts=%d frees=%d uploads=%d/%zu bytes downloads=%d/%zu bytes ShortConv=%d recurrence=%d sync=%d; owner 4 in {2,4,6}; no extra lease: PASS\n",
        allocs,frees,uploads,upload_bytes,downloads,download_bytes,shortconv_calls,recurrence_calls,sync_calls);
    puts("GLM53 CUDA recurrence production Segment integration: PASS");return 0;
}
