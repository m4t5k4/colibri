#include "cuda_kda_recur_ref.h"
#include "../delta_attention.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void cross_check(int heads, int dim, int zero) {
    size_t p=(size_t)heads*dim, ns=p*dim;
    float *cpu=malloc(ns*sizeof(float)), *ref=malloc(ns*sizeof(float));
    float *qkv=malloc(3*p*sizeof(float)), *mixed=malloc(3*p*sizeof(float));
    float *window=calloc(3*p,sizeof(float)), *taps=malloc(3*p*sizeof(float));
    float *decay=malloc(p*sizeof(float)), *beta=malloc((size_t)heads*sizeof(float));
    float *cpu_out=malloc(p*sizeof(float)), *ref_out=malloc(p*sizeof(float));
    float *scratch=malloc((size_t)coli_kda_scratch_floats(heads,dim,dim)*sizeof(float));
    assert(cpu && ref && qkv && mixed && window && taps && decay && beta && cpu_out && ref_out && scratch);
    kda_recur_initial(cpu,ns,zero); memcpy(ref,cpu,ns*sizeof(float));
    for (size_t i=0; i<3*p; i++) taps[i]=1.f;
    for (int step=0; step<64; step++) {
        kda_recur_inputs(qkv,qkv+p,qkv+2*p,decay,beta,heads,dim,dim,step);
        for (size_t i=0; i<3*p; i++) mixed[i]=coli_kda_silu(qkv[i]);
        assert(!coli_kda_step(cpu_out,cpu,window,qkv,taps,decay,beta,heads,dim,dim,1,1e-6f,scratch));
        kda_recur_ref(ref,ref_out,mixed,mixed+p,mixed+2*p,decay,beta,heads,dim,dim,1e-6f);
        assert(!memcmp(cpu,ref,ns*sizeof(float)) && !memcmp(cpu_out,ref_out,p*sizeof(float)));
    }
    printf("H=%d KD=VD=%d, %s initial state, 64 tokens: exact coli_kda_step state/raw output PASS\n",heads,dim,zero?"zero":"nonzero");
    free(cpu); free(ref); free(qkv); free(mixed); free(window); free(taps);
    free(decay); free(beta); free(cpu_out); free(ref_out); free(scratch);
}

/* Deliberately wrong variants prove the deterministic vectors reject the
 * listed semantic errors; these are not alternative production algorithms. */
enum { PRE_DECAY_MEMORY, NO_BETA, BETA_ON_STATE, NO_QUERY_SCALE, EPS_OUTSIDE,
       SWAP_NORMS, TRANSPOSE_STATE, HEAD_KEY_LAYOUT, WRONG_V_STRIDE, HEAD_DECAY, OLD_OUTPUT, MUTANTS };
static void mutant(int mode,float *state,float *out,const float *q,const float *k,
        const float *v,const float *decay,const float *beta,int heads,int kd,int vd,float eps) {
    for (int h=0; h<heads; h++) {
        const float *qh=q+(size_t)h*kd,*kh=k+(size_t)h*kd;
        float qs=0,ks=0;
        if (mode!=EPS_OUTSIDE) qs=ks=eps;
        for (int i=0; i<kd; i++) { qs+=qh[i]*qh[i]; ks+=kh[i]*kh[i]; }
        float qn=1.f/sqrtf(qs),kn=1.f/sqrtf(ks);
        if (mode==EPS_OUTSIDE) { qn=1.f/fmaxf(sqrtf(qs),eps); kn=1.f/fmaxf(sqrtf(ks),eps); }
        if (mode==SWAP_NORMS) { float temp=qn; qn=kn; kn=temp; }
        if (mode!=NO_QUERY_SCALE) qn/=sqrtf((float)kd);
        for (int j=0; j<vd; j++) {
            float memory=0,oldout=0;
            for (int i=0; i<kd; i++) {
                size_t ix=mode==HEAD_KEY_LAYOUT?((size_t)i*heads+h)*vd+j:
                    (size_t)h*kd*vd+(mode==TRANSPOSE_STATE?(size_t)j*kd+i:(size_t)i*vd+j);
                float old=state[ix],a=expf(decay[(size_t)h*kd+(mode==HEAD_DECAY?0:i)]);
                state[ix]*=a;
                if (mode==BETA_ON_STATE) state[ix]*=beta[h];
                memory+=(kh[i]*kn)*(mode==PRE_DECAY_MEMORY?old:state[ix]);
                oldout+=(qh[i]*qn)*state[ix];
            }
            float value=v[mode==WRONG_V_STRIDE?((size_t)h*(vd+1)+j)%((size_t)heads*vd):(size_t)h*vd+j];
            float y=0,b=(mode==NO_BETA||mode==BETA_ON_STATE)?1.f:beta[h];
            for (int i=0; i<kd; i++) {
                size_t ix=mode==HEAD_KEY_LAYOUT?((size_t)i*heads+h)*vd+j:
                    (size_t)h*kd*vd+(mode==TRANSPOSE_STATE?(size_t)j*kd+i:(size_t)i*vd+j);
                state[ix]+=(kh[i]*kn)*(value-memory)*b;
                y+=(qh[i]*qn)*state[ix];
            }
            out[(size_t)h*vd+j]=mode==OLD_OUTPUT?oldout:y;
        }
    }
}
static void mutation_check(void) {
    enum { H=3,KD=17,VD=11,N=H*KD*VD };
    float initial[N],expected[N],wrong[N],out[H*VD],badout[H*VD];
    float q[H*KD],k[H*KD],v[H*VD],decay[H*KD],beta[H];
    const char *names[]={"pre-decay prediction / decay after read","beta omitted","beta on state",
        "query scaling omitted","epsilon outside sqrt","query/key norms swapped",
        "state key/value layout transposed","state head/key layout transposed","value head stride","per-head decay","pre-correction output"};
    kda_recur_initial(initial,N,0);
    for (int mode=0; mode<MUTANTS; mode++) {
        int caught=0;
        for (int step=1; step<=12; step++) {
            kda_recur_inputs(q,k,v,decay,beta,H,KD,VD,step);
            memcpy(expected,initial,sizeof(initial)); memcpy(wrong,initial,sizeof(initial));
            kda_recur_ref(expected,out,q,k,v,decay,beta,H,KD,VD,1e-6f);
            mutant(mode,wrong,badout,q,k,v,decay,beta,H,KD,VD,1e-6f);
            for (int i=0; i<N; i++) caught |= fabsf(expected[i]-wrong[i])>1e-5f*(1.f+fabsf(expected[i]));
            for (int i=0; i<H*VD; i++) caught |= fabsf(out[i]-badout[i])>1e-5f*(1.f+fabsf(out[i]));
        }
        assert(caught); printf("reject %s: PASS\n",names[mode]);
    }
}
int main(void) {
    for (int zero=0; zero<=1; zero++) {
        cross_check(1,1,zero); cross_check(2,8,zero); cross_check(3,17,zero); cross_check(64,128,zero);
    }
    mutation_check(); puts("CUDA KDA recurrence CPU oracle: PASS"); return 0;
}
