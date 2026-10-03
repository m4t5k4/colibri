/* Range-local recurrent state, using production allocation without a checkpoint
 * or any GPU backend. Layer slots must retain their absolute model indices. */
#define GLM53_NO_MAIN
#include "../glm53.c"
#include <assert.h>

static void check_range(GModel *m, int begin, int end) {
    m->layer_begin = begin;
    m->layer_end = end;
    const Cfg *c = &m->c;
    const int cap = 7;
    GSession *s = session_open(m, cap);
    assert(s && s->layer && s->cap == cap && s->filled == 0);
    int kda = 0;
    for (int i = 0; i < c->n_layers; i++) {
        GLayerState *st = &s->layer[i];
        if (i < begin || i >= end) {
            assert(!st->latent && !st->ikeys && !st->igates);
            assert(!st->kda_state && !st->kda_window);
        } else if (c->is_full[i]) {
            assert(st->latent && st->ikeys && st->igates);
            assert(!st->kda_state && !st->kda_window);
            /* Touch the last allocated cell at its absolute layer index. */
            st->latent[(size_t)cap * c->kv_lora - 1] = (float)i;
            st->ikeys[(size_t)cap * c->index_hd - 1] = (float)i;
            st->igates[(size_t)cap * c->index_hd - 1] = (float)i;
        } else {
            assert(st->kda_state && st->kda_window);
            assert(!st->latent && !st->ikeys && !st->igates);
            const size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd;
            const size_t nw = (size_t)3 * c->kda_proj * c->conv_k;
            for (size_t j = 0; j < ns; j++) assert(st->kda_state[j] == 0.0f);
            for (size_t j = 0; j < nw; j++) assert(st->kda_window[j] == 0.0f);
            st->kda_state[ns - 1] = (float)i;
            st->kda_window[nw - 1] = (float)i;
            kda++;
        }
    }
    assert((s->kda_scratch != NULL) == (kda > 0));

    /* Segment snapshots already describe exactly the same owned state. */
    Glm53SegmentEngine engine = {0};
    engine.model = *m;
    engine.layer_begin = (uint32_t)begin;
    engine.layer_end = (uint32_t)end;
    Glm53SegmentSession segment = {.engine = &engine, .session = s,
                                  .context_tokens = (uint32_t)cap};
    ColiSegmentStateSpan spans[GLM53_SEGMENT_MAX_SPANS];
    const size_t count = glm53_segment_spans(&engine, &segment, spans,
                                            GLM53_SEGMENT_MAX_SPANS);
    size_t at = 0;
    for (int i = begin; i < end; i++) {
        GLayerState *st = &s->layer[i];
        if (c->is_full[i]) {
            assert(spans[at++].data == st->latent);
            assert(spans[at++].data == st->ikeys);
            assert(spans[at++].data == st->igates);
        } else {
            assert(spans[at++].data == st->kda_state);
            assert(spans[at++].data == st->kda_window);
        }
    }
    assert(at == count);
    session_close(m, s);
    session_close(m, NULL);
    printf("session range [%d, %d): absolute slots=%d, spans=%zu, KDA scratch=%s: OK\n",
           begin, end, c->n_layers, count, kda ? "yes" : "no");
}

int main(void) {
    GModel m = {0};
    m.c = (Cfg){.n_layers = 6, .kv_lora = 3, .index_hd = 2,
                .kda_heads = 2, .kda_hd = 4, .kda_proj = 8, .conv_k = 4};
    m.c.is_full[1] = m.c.is_full[3] = m.c.is_full[5] = 1;
    check_range(&m, 2, 4);  /* Owned KDA and MLA; both types outside too. */
    check_range(&m, 3, 4);  /* MLA only, despite global KDA configuration. */
    check_range(&m, 4, 5);  /* KDA only, at an absolute nonzero index. */
    check_range(&m, 3, 3);  /* Empty range retains all metadata slots. */
    check_range(&m, 6, 6);  /* Empty range at the model boundary. */
    check_range(&m, 0, 6);  /* Full-model allocation remains unchanged. */
    puts("GLM53 range-native session state: OK");
    return 0;
}
