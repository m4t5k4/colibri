/* Test-local full-K oldest-first ShortConv oracle, independent of GLM53. */
#ifndef TEST_CUDA_KDA_SHORTCONV_REF_H
#define TEST_CUDA_KDA_SHORTCONV_REF_H
#include <math.h>
#include <stddef.h>
#include <string.h>

static void kda_shortconv_ref(float *window, float *mixed, const float *qkv,
        const float *conv, int channels, int kernel) {
    for (int channel=0; channel<channels; channel++) {
        float *history=window+(size_t)channel*kernel;
        memmove(history,history+1,(size_t)(kernel-1)*sizeof(float));
        history[kernel-1]=qkv[channel];
        float sum=0.f;
        for (int tap=0; tap<kernel; tap++) sum+=conv[(size_t)channel*kernel+tap]*history[tap];
        mixed[channel]=sum/(1.f+expf(-sum));
    }
}
static float kda_shortconv_input(int channel,int step) {
    return (float)(((size_t)channel*7+(size_t)step*5)%31)*0.041f-15.f*0.041f;
}
static void kda_shortconv_initial(float *window,float *conv,int channels,int kernel) {
    for (int c=0; c<channels; c++) for (int tap=0; tap<kernel; tap++) {
        size_t at=(size_t)c*kernel+tap;
        window[at]=(float)((int)(((size_t)c*11+(size_t)tap*3)%23)-11)*0.023f;
        conv[at]=(float)((int)(((size_t)c*3+(size_t)tap*5)%19)-9)*0.037f;
    }
}
#endif
