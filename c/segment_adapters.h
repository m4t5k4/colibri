#ifndef COLIBRI_SEGMENT_ADAPTERS_H
#define COLIBRI_SEGMENT_ADAPTERS_H
#include <stdint.h>

/* Optional GLM53 Segment resource_plan, native-endian fixed-width prefix.
 * struct_size includes future extension bytes and must fit resource_plan_size.
 * COLI_GPU(S) configures the complete process list; this ordinal is the sole
 * logical stage owner, never a list index. No plan leaves the engine unplanned.
 * A supplied malformed plan is rejected. Planned engines require COLI_CUDA=1
 * and a CUDA-enabled build; this plan enables no CUDA execution kernels. */
#define COLI_GLM53_STAGE_PLAN_VERSION 1u
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    int32_t cuda_device_ordinal;
} ColiGlm53StagePlan;

/*
 * Explicit registration entry points for Colibri's built-in model adapters.
 *
 * Each engine owns its implementation and is normally linked as a separate
 * executable/object.  A Segment consumer links the engines it wants and calls
 * the matching functions before the first adapter lookup.  The ordinary
 * Colibri CLI/server paths do not call these functions, so adding an adapter
 * cannot change standalone inference or initialization order.
 */

#ifdef __cplusplus
extern "C" {
#endif

int coli_glm_segment_adapter_register(void);
int coli_glm53_segment_adapter_register(void);
int coli_inkling_segment_adapter_register(void);
int coli_kimi_segment_adapter_register(void);
int coli_olmoe_segment_adapter_register(void);
int coli_qwen36_segment_adapter_register(void);
int coli_qwen38_segment_adapter_register(void);
int coli_deepseek_v4_segment_adapter_register(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_SEGMENT_ADAPTERS_H */
