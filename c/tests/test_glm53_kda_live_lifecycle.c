/* Real-model lifecycle driver. Production internals are included, not edited.
 * The separately linked backend wrapper is the only fault-injection site. */
#define GLM53_NO_MAIN
#include "../glm53.c"
#include "glm53_kda_live_test_api.h"

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line=%d condition=%s\n", __LINE__, #x); exit(1); \
} } while (0)

static size_t ns, nw;
static int nkda;
static float *last_logits;
/* Fail-closed integration thresholds, not claims of measured rig accuracy.
 * Report actual errors; do not relax these automatically on a failing run. */
static void compare(const char *label, const float *a, const float *b, size_t n,
                    int exact) {
    double absolute = 0, relative = 0;
    size_t index = 0;
    for (size_t i = 0; i < n; i++) {
        REQUIRE(isfinite(a[i]) && isfinite(b[i]));
        double e = fabs((double)a[i] - b[i]);
        double r = e / fmax(fabs((double)b[i]), 1e-12);
        if (e > absolute) { absolute = e; index = i; }
        if (r > relative) relative = r;
        REQUIRE(exact ? e == 0 : e <= 1e-4 + 1e-4 * fabs((double)b[i]));
    }
    printf("comparison=%s exact=%d max_abs=%.9g max_rel=%.9g max_abs_index=%zu pass=1\n",
           label, exact, absolute, relative, index);
}
static unsigned long long digest(const float *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 14695981039346656037ULL;
    for (size_t i = 0; i < n * sizeof(float); i++) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
static void health(GModel *m) {
    REQUIRE(m->cuda.active && !m->cuda.failed && m->cuda.errors == 0);
    printf("expert_health errors=%llu uploads=%llu resident=%u cuda_rows=%llu fallback_rows=%llu\n",
        (unsigned long long)m->cuda.errors, (unsigned long long)m->cuda.uploads,
        m->cuda.resident, (unsigned long long)m->cuda.executed,
        (unsigned long long)m->cuda.fallback);
}
static void counters(GModel *m, const char *where) {
    G53KdaCudaTier *t = &m->kda_tier;
    printf("boundary=%s kda_cuda_calls=%llu kda_cuda_errors=%llu kda_cuda_fallbacks=%llu state_pushes=%llu state_pulls=%llu invalidations=%llu\n",
        where, (unsigned long long)t->calls, (unsigned long long)t->errors,
        (unsigned long long)t->fallbacks, (unsigned long long)t->pushes,
        (unsigned long long)t->pulls, (unsigned long long)t->invalidations);
    health(m);
}
static void compare_state(GModel *m, GSession *s, GSession *ref) {
    float *state = malloc(ns * sizeof(float)), *window = malloc(nw * sizeof(float));
    REQUIRE(state && window);
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_full[i]) {
        const float *a = s->layer[i].kda_state, *w = s->layer[i].kda_window;
        G53KdaCudaLayer *e = &m->kda_gpu[i];
        if (e->owner == s && e->host_stale) {
            /* Observation does not update host freshness: pin/recovery must
             * still perform its own production materialization. */
            REQUIRE(coli_cuda_kda_get_state(e->object, state, window));
            a = state; w = window;
        }
        char label[80];
        snprintf(label, sizeof(label), "layer%d_state", i);
        compare(label, a, ref->layer[i].kda_state, ns, 0);
        snprintf(label, sizeof(label), "layer%d_window", i);
        compare(label, w, ref->layer[i].kda_window, nw, 0);
    }
    free(state); free(window);
}
static int logits_equal(GModel *m, const float *a, const float *b, const char *where) {
    REQUIRE(a && b);
    for (int v = 0; v < m->c.vocab; v++) REQUIRE(isfinite(a[v]) && isfinite(b[v]));
    int token = argmax(a, m->c.vocab);
    REQUIRE(token == argmax(b, m->c.vocab));
    printf("semantic=%s live_token=%d reference_token=%d pass=1\n", where, token, token);
    return token;
}
static float *cpu_decode(GModel *m, GSession *s, int id) {
    int enabled = m->kda_tier.enabled; m->kda_tier.enabled = 0;
    float *out = forward_decode(m, s, &id);
    m->kda_tier.enabled = enabled;
    return out;
}
static int paired_token(GModel *m, KVSlot *slot, GSession *ref, int *ids,
                        int id, const char *where) {
    float *want = cpu_decode(m, ref, id);
    float *got = forward_decode(m, slot->session, &id);
    ids[slot->session->filled - 1] = id;
    slot_remember(slot, ids, slot->session->filled);
    int next = logits_equal(m, got, want, where);
    compare_state(m, slot->session, ref);
    free(last_logits); last_logits = got;
    free(want); health(m);
    return next;
}
static int start(GModel *m, KVSlot *slot, GSession **ref, int *ids, float **logits) {
    slot->session = session_open(m, 128); *ref = session_open(m, 128);
    /* Real tokenizer and weights, no fabricated hidden inputs. */
    Tok tok; char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.json", getenv("SNAP"));
    tok_load(&tok, path);
    const char *prompt = "Explain virtual memory and page faults briefly.";
    int n = tok_encode(&tok, prompt, (int)strlen(prompt), ids, 96);
    tok_free(&tok); REQUIRE(n > 1 && n < 96);
    int enabled = m->kda_tier.enabled; m->kda_tier.enabled = 0;
    float *want = forward_prefill(m, *ref, ids, n, NULL, 0, 0);
    m->kda_tier.enabled = enabled;
    *logits = forward_prefill(m, slot->session, ids, n, NULL, 0, 0);
    slot_remember(slot, ids, n);
    int next = logits_equal(m, *logits, want, "prefill");
    free(want);
    compare_state(m, slot->session, *ref);
    return next;
}
static void drop(GModel *m, KVSlot *slot, GSession *ref) {
    free(last_logits); last_logits = NULL;
    slot_reset(m, slot); session_close(m, ref); free(slot->tokens);
    for (int i = 0; i < COLI_PIN_SLOTS_MAX; i++) {
        free(slot->pins.slot[i].ids); free(slot->pins.slot[i].logit);
    }
}
static void pin_case(GModel *m) {
    KVSlot slot = {0}; GSession *ref; int ids[128]; float *logits;
    int next = start(m, &slot, &ref, ids, &logits); free(logits);
    next = paired_token(m, &slot, ref, ids, next, "A_success1");
    next = paired_token(m, &slot, ref, ids, next, "A_success2");
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_full[i])
        REQUIRE(m->kda_gpu[i].owner == slot.session && m->kda_gpu[i].host_stale);
    uint64_t pulls = m->kda_tier.pulls;
    int saved = slot.session->filled;
    REQUIRE(slot_pin_save(m, &slot, ids, saved, last_logits));
    REQUIRE(m->kda_tier.pulls == pulls + (uint64_t)nkda);
    ids[saved] = next;
    int pi = coli_pin_best(&slot.pins, ids, saved + 1); REQUIRE(pi >= 0);
    Glm53PinState *pin = slot.pins.slot[pi].state;
    compare("pin_logits_copy", last_logits, slot.pins.slot[pi].logit, m->c.vocab, 1);
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_full[i]) {
        REQUIRE(!m->kda_gpu[i].host_stale);
        compare("pin_state_copy", slot.session->layer[i].kda_state, pin->state[i], ns, 1);
        compare("pin_window_copy", slot.session->layer[i].kda_window, pin->window[i], nw, 1);
    }
    counters(m, "A_saved_from_stale");
    uint64_t calls = m->kda_tier.calls;
    for (int j = 0; j < 2; j++) {
        int token = next; float *out = forward_decode(m, slot.session, &token);
        ids[slot.session->filled - 1] = token; next = argmax(out, m->c.vocab); free(out);
        slot_remember(&slot, ids, slot.session->filled);
    }
    REQUIRE(slot.session->filled == saved + 2);
    REQUIRE(m->kda_tier.calls == calls + 2u * nkda);
    int first = 0; while (m->c.is_full[first]) first++;
    float *advanced_state = malloc(ns * sizeof(float)), *advanced_window = malloc(nw * sizeof(float));
    REQUIRE(advanced_state && advanced_window && m->kda_gpu[first].host_stale);
    REQUIRE(coli_cuda_kda_get_state(m->kda_gpu[first].object, advanced_state, advanced_window));
    REQUIRE(memcmp(advanced_state, pin->state[first], ns * sizeof(float)) != 0);
    REQUIRE(memcmp(advanced_window, pin->window[first], nw * sizeof(float)) != 0);
    free(advanced_state); free(advanced_window);
    uint64_t invalidations = m->kda_tier.invalidations;
    pulls = m->kda_tier.pulls;
    REQUIRE(slot_pin_restore(m, &slot, ids, slot.n) == saved);
    REQUIRE(m->kda_tier.pulls == pulls);
    REQUIRE(m->kda_tier.invalidations == invalidations + (uint64_t)nkda);
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_full[i]) {
        REQUIRE(!m->kda_gpu[i].valid && !m->kda_gpu[i].host_stale);
        compare("restore_state_exact", slot.session->layer[i].kda_state, pin->state[i], ns, 1);
        compare("restore_window_exact", slot.session->layer[i].kda_window, pin->window[i], nw, 1);
    }
    compare_state(m, slot.session, ref);
    next = argmax(slot.pins.slot[pi].logit, m->c.vocab);
    REQUIRE(next == ids[saved]); /* Independent reference's continuation token. */
    uint64_t pushes = m->kda_tier.pushes;
    for (int j = 0; j < 3; j++) next = paired_token(m, &slot, ref, ids, next, "A_restored_continue");
    REQUIRE(m->kda_tier.pushes == pushes + (uint64_t)nkda);
    REQUIRE(m->kda_tier.errors == 0 && m->kda_tier.fallbacks == 0);
    counters(m, "A_complete"); drop(m, &slot, ref);
    puts("case=A_STALE_PIN_RESTORE pass=1");
}
static void failure_case(GModel *m) {
    KVSlot slot = {0}; GSession *ref; int ids[128]; float *logits;
    int next = start(m, &slot, &ref, ids, &logits); free(logits);
    next = paired_token(m, &slot, ref, ids, next, "B_success1");
    next = paired_token(m, &slot, ref, ids, next, "B_success2");
    int target = 0; while (m->c.is_full[target]) target++;
    REQUIRE(m->kda_gpu[target].host_stale && m->kda_gpu[target].valid);
    uint64_t calls = m->kda_tier.calls, pulls = m->kda_tier.pulls;
    uint64_t pushes = m->kda_tier.pushes, invalidations = m->kda_tier.invalidations;
    REQUIRE(g53_live_arm(m->kda_gpu[target].object));
    next = paired_token(m, &slot, ref, ids, next, "B_injected_token");
    G53LiveEvidence ev = g53_live_evidence();
    REQUIRE(ev.injected && ev.committed_generation_unchanged &&
            ev.committed_state_unchanged && ev.committed_window_unchanged &&
            ev.next_state_mutated && ev.next_window_mutated && ev.recovery_pull_exact);
    REQUIRE(m->kda_tier.errors == 1 && m->kda_tier.fallbacks == 1);
    REQUIRE(m->kda_tier.calls == calls + (uint64_t)nkda - 1);
    REQUIRE(m->kda_tier.pulls == pulls + 1);
    REQUIRE(m->kda_tier.pushes == pushes && m->kda_tier.invalidations == invalidations + 1);
    REQUIRE(!m->kda_gpu[target].usable && !m->kda_gpu[target].host_stale && !m->kda_gpu[target].valid);
    /* Independent local CPU reference from exact committed device state and
     * actual failed-token input proves recovery applies this token once. */
    GSession local = {0}; local.layer = calloc((size_t)m->c.n_layers, sizeof(*local.layer));
    local.kda_scratch = malloc((size_t)coli_kda_scratch_floats(m->c.kda_heads,
        m->c.kda_hd, m->c.kda_hd) * sizeof(float));
    float *s = malloc(ns * sizeof(float)), *w = malloc(nw * sizeof(float));
    float *x = malloc((size_t)m->c.hidden * sizeof(float)), *out = malloc((size_t)m->c.hidden * sizeof(float));
    REQUIRE(local.layer && local.kda_scratch && s && w && x && out);
    REQUIRE(g53_live_failure_input(s, w, x));
    printf("target_layer=%d committed_state_fnv64=%016llx committed_window_fnv64=%016llx\n",
        target, digest(s, ns), digest(w, nw));
    local.layer[target].kda_state = s; local.layer[target].kda_window = w;
    GModel cpu = *m; cpu.kda_tier.enabled = 0;
    kda_layer(&cpu, &cpu.c, &cpu.layer[target], x, 1, out, &local, target);
    compare("failed_token_cpu_once_state", slot.session->layer[target].kda_state, s, ns, 1);
    compare("failed_token_cpu_once_window", slot.session->layer[target].kda_window, w, nw, 1);
    printf("target_layer=%d recovered_state_fnv64=%016llx recovered_window_fnv64=%016llx committed_generation_unchanged=1 committed_state_unchanged=1 committed_window_unchanged=1 next_state_mutated=1 next_window_mutated=1 recovery_pull_exact=1 cpu_token_once=1\n",
        target, digest(s, ns), digest(w, nw));
    free(s); free(w); free(x); free(out); free(local.layer); free(local.kda_scratch);
    calls = m->kda_tier.calls;
    uint64_t uploads = m->cuda.uploads, rows = m->cuda.executed;
    for (int j = 0; j < 3; j++) next = paired_token(m, &slot, ref, ids, next, "B_cpu_recovery_continue");
    REQUIRE(m->kda_tier.calls == calls + 3u * (nkda - 1));
    REQUIRE(m->kda_tier.errors == 1 && m->kda_tier.fallbacks == 4);
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_full[i] && i != target)
        REQUIRE(m->kda_gpu[i].usable && m->kda_gpu[i].host_stale);
    REQUIRE(m->cuda.executed > rows);
    printf("expert_continuation uploads_before=%llu uploads_after=%llu uploads_delta=%llu executed_delta=%llu\n",
        (unsigned long long)uploads, (unsigned long long)m->cuda.uploads,
        (unsigned long long)(m->cuda.uploads - uploads), (unsigned long long)(m->cuda.executed - rows));
    counters(m, "B_complete_expected_errors1_fallbacks4");
    g53_live_clear(); drop(m, &slot, ref);
    puts("case=B_FAILURE_AFTER_SUCCESS pass=1 expected_injections=1 unexpected_errors=0");
}
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    REQUIRE(argc == 2);
    REQUIRE(getenv("GLM53_CUDA_KDA") && !strcmp(getenv("GLM53_CUDA_KDA"), "1"));
    REQUIRE(getenv("GLM53_EXPERT_GB") && !strcmp(getenv("GLM53_EXPERT_GB"), "8"));
    REQUIRE(getenv("USAGE_SAVE") && !strcmp(getenv("USAGE_SAVE"), "0"));
    REQUIRE(getenv("GLM53_CUDA_WARM_RESIDENCY") && !strcmp(getenv("GLM53_CUDA_WARM_RESIDENCY"), "0"));
    REQUIRE(getenv("COLI_USAGE"));
    setenv("SNAP", argv[1], 1); setenv("COLI_PIN_SLOTS", "4", 1);
    GModel m = {0}; model_load(&m, argv[1]); glm53_telemetry_init(argv[1], &m.c);
    REQUIRE(m.kda_tier.enabled && m.kda_gpu && m.kda_tier.ndev == 8);
    ns = (size_t)m.c.kda_heads * m.c.kda_hd * m.c.kda_hd;
    nw = (size_t)3 * m.c.kda_proj * m.c.conv_k;
    for (int i = 0; i < m.c.n_layers; i++) if (!m.c.is_full[i]) {
        REQUIRE(m.kda_gpu[i].usable && m.kda_gpu[i].object);
        REQUIRE(m.kda_gpu[i].device == m.kda_tier.devices[nkda % m.kda_tier.ndev]); nkda++;
    }
    REQUIRE(nkda == 34); health(&m);
    pin_case(&m); failure_case(&m);
    glm53_kda_cuda_close(&m); model_release(&m);
    puts("PASS real CUDA GLM53 pin/restore and transactional CPU recovery; expected_injections=1");
    return 0;
}
