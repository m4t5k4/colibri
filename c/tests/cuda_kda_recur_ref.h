/* Test-local scalar recurrence oracle; no backend or model dependency. */
#ifndef TEST_CUDA_KDA_RECUR_REF_H
#define TEST_CUDA_KDA_RECUR_REF_H
#include <math.h>
#include <stddef.h>
#include <string.h>

static void kda_recur_ref(float *state, float *out, const float *q,
        const float *key, const float *value, const float *decay,
        const float *beta, int heads, int kd, int vd, float eps) {
    for (int h=0; h<heads; h++) {
        const float *qh=q+(size_t)h*kd, *kh=key+(size_t)h*kd;
        float *s=state+(size_t)h*kd*vd, *y=out+(size_t)h*vd;
        float qs=eps, ks=eps, memory[256];
        for (int i=0; i<kd; i++) { qs+=qh[i]*qh[i]; ks+=kh[i]*kh[i]; }
        float scale=1.f/sqrtf((float)kd), qnorm=scale/sqrtf(qs), knorm=1.f/sqrtf(ks);
        memset(memory,0,(size_t)vd*sizeof(float));
        for (int i=0; i<kd; i++) {
            float alpha=expf(decay[(size_t)h*kd+i]), sk=kh[i]*knorm;
            for (int j=0; j<vd; j++) {
                s[(size_t)i*vd+j]*=alpha;
                memory[j]+=sk*s[(size_t)i*vd+j];
            }
        }
        memset(y,0,(size_t)vd*sizeof(float));
        for (int i=0; i<kd; i++) {
            float sk=kh[i]*knorm, sq=qh[i]*qnorm;
            for (int j=0; j<vd; j++) {
                /* Match delta_attention.h's actual left-associated expression. */
                s[(size_t)i*vd+j]+=sk*(value[(size_t)h*vd+j]-memory[j])*beta[h];
                y[j]+=sq*s[(size_t)i*vd+j];
            }
        }
    }
}
static void kda_recur_inputs(float *q, float *k, float *v, float *decay,
        float *beta, int heads, int kd, int vd, int step) {
    for (int h=0; h<heads; h++) {
        beta[h]=step%11==0 ? 0.f : step%7==0 ? 1.f : (float)((h+step)%9)/8.f;
        for (int i=0; i<kd; i++) {
            size_t p=(size_t)h*kd+i;
            q[p]=(float)((h*7+i*3+step*5)%23-11)*0.031f;
            k[p]=(float)((h*3+i*7+step*2)%19-9)*0.047f;
            if (step%9==1 && h==0) q[p]*=1e-7f;
            if (step%9==2 && h==0) k[p]*=1e-7f;
            decay[p]=step%5==0 ? 0.f : -(float)((h+i*3+step)%13)*0.021f;
        }
        for (int j=0; j<vd; j++)
            v[(size_t)h*vd+j]=(float)((h*5+j*11+step*3)%29-14)*0.023f;
    }
}
static void kda_recur_initial(float *state, size_t n, int zero) {
    for (size_t i=0; i<n; i++) state[i]=zero ? 0.f : (float)((int)(i%31)-15)*0.009f;
}
#endif
