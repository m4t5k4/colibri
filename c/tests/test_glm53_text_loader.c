/* Intercept actual payload reads without changing the production loader. */
#define _GNU_SOURCE
#include "../st.h"
#include <assert.h>

static size_t visual_reads, text_reads;
static int64_t counted_read(shards *s, const char *name, float *out,
                            int64_t cap, int drop) {
    if (!strncmp(name, "model.visual.", 13)) visual_reads++;
    else text_reads++;
    return st_read_f32_cap(s, name, out, cap, drop);
}
#define st_read_f32_cap counted_read
#define GLM53_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#define COLI_EDGE_ADAPTER
#include "../glm53.c"
#undef st_read_f32_cap

static void unloaded(const GModel *m) {
    const ColiVisionTower zero = {0};
    assert(m->c.vis_layers == 1); /* Metadata survives the opt-out. */
    assert(!m->has_vision && !m->vblocks);
    assert(!memcmp(&m->vision, &zero, sizeof(zero)));
    assert(visual_reads == 0 && text_reads > 0);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    char error[256] = {0};
    void *impl = NULL;
    ColiEdgeCapabilities edge;
    ColiEdgeEngineOptions eo = {.model_dir = argv[1]};
    assert(!glm53_edge_engine_open(&impl, &edge, &eo, error, sizeof(error)));
    Glm53EdgeEngine *ee = impl;
    unloaded(&ee->model);
    assert(ee->model.embed && ee->model.final_norm && ee->model.head.f);
    assert(edge.state_width == 4 && edge.vocab_size == 8);
    assert(edge.flags == (COLI_EDGE_CAP_CPU | COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS));
    assert(!strcmp(edge.state_schema, "glm53/mla-latent-kda-conv-dsa-f32-v1"));
    printf("Edge: text=%zu vision=%zu\n", text_reads, visual_reads);
    glm53_edge_engine_destroy(impl);

    visual_reads = text_reads = 0;
    ColiSegmentCapabilities segment;
    ColiSegmentEngineOptions so = {.model_dir = argv[1], .layer_end = 1,
                                   .context_tokens = 4};
    assert(!glm53_segment_engine_open(&impl, &segment, &so, error, sizeof(error)));
    Glm53SegmentEngine *se = impl;
    unloaded(&se->model);
    assert(se->model.layer[0].kq.f && !se->model.embed);
    assert(segment.state_width == edge.state_width);
    assert(segment.flags == (COLI_SEGMENT_CAP_CPU | COLI_SEGMENT_CAP_SNAPSHOT |
                            COLI_SEGMENT_CAP_RANGE_NATIVE | COLI_SEGMENT_CAP_MULTI_SESSION));
    assert(!strcmp(segment.state_schema, edge.state_schema));
    assert(!strcmp(segment.numeric_class, edge.numeric_class));
    printf("Segment: text=%zu vision=%zu\n", text_reads, visual_reads);
    glm53_segment_engine_destroy(impl);

    visual_reads = text_reads = 0;
    GModel normal = {0};
    model_load(&normal, argv[1]);
    assert(normal.has_vision && normal.vblocks && normal.vision.patch_w);
    assert(visual_reads == 25 && text_reads > 0);
    printf("Normal: text=%zu vision=%zu\n", text_reads, visual_reads);
    model_release(&normal);
    model_release(&normal); /* Released state remains safe. */

    /* The old adapter ranges used the vision-enabled compatibility wrapper. */
    GModel old_edge = {0}, old_segment = {0};
    visual_reads = text_reads = 0;
    model_load_range(&old_edge, argv[1], 0, 0, 1);
    assert(visual_reads == 25);
    model_release(&old_edge);
    visual_reads = text_reads = 0;
    model_load_range(&old_segment, argv[1], 0, 1, 0);
    assert(visual_reads == 25);
    model_release(&old_segment);
    puts("Before: Edge=25 Segment=25; after: Edge=0 Segment=0; normal=25: OK");
    return 0;
}
