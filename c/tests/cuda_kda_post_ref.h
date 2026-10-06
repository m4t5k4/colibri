#ifndef CUDA_KDA_POST_REF_H
#define CUDA_KDA_POST_REF_H
#include <math.h>
#include <stddef.h>
static inline void kda_post_ref(float *core, const float *gate, const float *onorm,
        int heads, int dim, float eps) {
    for (int h=0; h<heads; h++) {
        size_t base=(size_t)h*dim;
        float square=0.f;
        for(int d=0;d<dim;d++) square+=core[base+d]*core[base+d];
        float inverse=1.f/sqrtf(square/dim+eps);
        for(int d=0;d<dim;d++) {
            float g=gate[base+d];
            float sigmoid=g>=0.f?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
            core[base+d]=core[base+d]*inverse*onorm[d]*sigmoid;
        }
    }
}
#endif
