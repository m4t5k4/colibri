#include "cuda_kda_shortconv_ref.h"
#include "../delta_attention.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void ordering_probe(void) {
    float window[]={1,2,4,8},qkv[]={16},taps[]={0.125f,-0.25f,0.5f,0.0625f},mixed[1];
    const float expected[]={2,4,8,16};
    kda_shortconv_ref(window,mixed,qkv,taps,1,4);
    assert(!memcmp(window,expected,sizeof(expected)));
    assert(mixed[0]==coli_kda_silu(4.25f));
    assert(fabsf(mixed[0]-coli_kda_silu(2.625f))>0.1f); /* unshifted old samples */
    assert(fabsf(mixed[0]-coli_kda_silu(2.125f))>0.1f); /* reversed logical history */
    float one[]={-3},current[]={2},weight[]={-0.75f};
    kda_shortconv_ref(one,mixed,current,weight,1,1);
    assert(one[0]==2 && mixed[0]==coli_kda_silu(-1.5f));
    puts("explicit [1,2,4,8] -> [2,4,8,16], increasing asymmetric taps and K=1: PASS");
}
static void sequence(int channels,int kernel) {
    size_t n=(size_t)channels*kernel;
    float *window=malloc(n*sizeof(float)),*initial=malloc(n*sizeof(float)),*taps=malloc(n*sizeof(float));
    float *qkv=malloc((size_t)channels*sizeof(float)),*mixed=malloc((size_t)channels*sizeof(float));
    assert(window && initial && taps && qkv && mixed);
    kda_shortconv_initial(window,taps,channels,kernel);memcpy(initial,window,n*sizeof(float));
    for (int step=0; step<64; step++) {
        for (int c=0; c<channels; c++) qkv[c]=kda_shortconv_input(c,step);
        kda_shortconv_ref(window,mixed,qkv,taps,channels,kernel);
        for (int c=0; c<channels; c++) {
            float sum=0.f;
            for (int tap=0; tap<kernel; tap++) {
                int token=step-(kernel-1-tap);
                float expected=token>=0?kda_shortconv_input(c,token):initial[(size_t)c*kernel+tap+step+1];
                assert(!memcmp(&window[(size_t)c*kernel+tap],&expected,sizeof(float)));
                sum+=taps[(size_t)c*kernel+tap]*expected;
            }
            assert(mixed[c]==coli_kda_silu(sum));
        }
    }
    printf("channels=%d K=%d, 64 tokens: exact full window and scalar SiLU PASS\n",channels,kernel);
    free(window);free(initial);free(taps);free(qkv);free(mixed);
}
static void cross_check(int heads,int dim,int kernel) {
    int p=heads*dim,channels=3*p;size_t n=(size_t)channels*kernel;
    float *a=malloc(n*sizeof(float)),*b=malloc(n*sizeof(float)),*conv=malloc(n*sizeof(float));
    float *qkv=malloc((size_t)channels*sizeof(float)),*mixed=malloc((size_t)channels*sizeof(float));
    float *scratch=malloc((size_t)coli_kda_scratch_floats(heads,dim,dim)*sizeof(float));
    float *state=calloc((size_t)p*dim,sizeof(float)),*out=malloc((size_t)p*sizeof(float));
    float *decay=calloc((size_t)p,sizeof(float)),*beta=calloc((size_t)heads,sizeof(float));
    assert(a && b && conv && qkv && mixed && scratch && state && out && decay && beta);
    kda_shortconv_initial(a,conv,channels,kernel);memcpy(b,a,n*sizeof(float));
    for (int step=0; step<64; step++) {
        for (int c=0; c<channels; c++) qkv[c]=kda_shortconv_input(c,step);
        kda_shortconv_ref(a,mixed,qkv,conv,channels,kernel);
        assert(!coli_kda_step(out,state,b,qkv,conv,decay,beta,heads,dim,dim,kernel,1e-6f,scratch));
        assert(!memcmp(a,b,n*sizeof(float)) && !memcmp(mixed,scratch,(size_t)channels*sizeof(float)));
    }
    printf("coli_kda_step H=%d D=%d K=%d: 64-token exact window/mixed cross-check PASS\n",heads,dim,kernel);
    free(a);free(b);free(conv);free(qkv);free(mixed);free(scratch);free(state);free(out);free(decay);free(beta);
}
int main(void) {
    ordering_probe();sequence(1,1);sequence(2,2);sequence(7,4);sequence(257,4);sequence(24576,4);sequence(7,7);
    cross_check(1,1,1);cross_check(1,2,2);cross_check(1,7,4);cross_check(1,7,7);cross_check(64,128,4);
    puts("CUDA KDA ShortConv CPU oracle: PASS");return 0;
}
