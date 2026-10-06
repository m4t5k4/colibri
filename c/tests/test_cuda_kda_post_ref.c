#include "cuda_kda_post_ref.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const int shapes[][2]={{1,1},{2,3},{3,17},{64,128},{2,256}};
    for(size_t j=0;j<sizeof(shapes)/sizeof(shapes[0]);j++) {
        int h=shapes[j][0],d=shapes[j][1];size_t n=(size_t)h*d;
        float *c=malloc(n*4),*g=malloc(n*4),*norm=malloc((size_t)d*4),*expected=malloc(n*4);
        assert(c && g && norm && expected);
        for(int x=0;x<d;x++)norm[x]=0.7f+(float)(x%7)*0.09f;
        for(size_t x=0;x<n;x++){c[x]=(float)((int)(x%23)-11)*0.043f;g[x]=(float)((int)(x%31)-15)*0.37f;}
        if(n>1){g[0]=-100.f;g[n-1]=100.f;}
        for(int head=0;head<h;head++) {
            float square=0.f;for(int x=0;x<d;x++)square+=c[(size_t)head*d+x]*c[(size_t)head*d+x];
            float inverse=1.f/sqrtf(square/d+0.003f);
            for(int x=0;x<d;x++){size_t at=(size_t)head*d+x;float sig=g[at]>=0.f?1.f/(1.f+expf(-g[at])):expf(g[at])/(1.f+expf(g[at]));expected[at]=c[at]*inverse*norm[x]*sig;}
        }
        kda_post_ref(c,g,norm,h,d,0.003f);
        for(size_t x=0;x<n;x++)assert(isfinite(c[x]) && fabsf(c[x]-expected[x])<=1e-5f*(1.f+fabsf(expected[x])));
        printf("KDA post H=%d D=%d nonuniform onorm/gate, scaled 1e-5: PASS\n",h,d);
        free(c);free(g);free(norm);free(expected);
    }
    return 0;
}
