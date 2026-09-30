#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
test -x c/glm53
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export COLI_CUDA=1 COLI_GPU= COLI_GPUS=0,1,2,3,4,5,6,7 CUDA_EXPERT_GB=48
export GLM53_CUDA_HEAT_MIN=2 GLM53_CUDA_HEAT_MARGIN=1
export GLM53_CUDA_PARALLEL_PROMOTE=1 GLM53_CUDA_PROMOTE_OVERLAP=1 GLM53_CUDA_PROMOTE_LATE_JOIN=1
export GLM53_MLA_OUT_ROWS4=1 GLM53_MLA_QB_ROWS4=0 GLM53_MLA_ABSORBED_BATCH=0
export GLM53_CUDA_WARM_RESIDENCY=0 GLM53_CUDA_PROFILE=1 GLM53_VERBOSE=1
export GLM53_MAXT=1024 DRAFT=0 USAGE_SAVE=0
export COLI_USAGE="$PWD/artifacts/phase2e/usage-snapshot-before-phase2e"
mkdir -p artifacts/phase2f-live
run=$(mktemp -d "$PWD/artifacts/phase2f-live/serve.XXXXXX")
fingerprint() {
    git rev-parse HEAD
    git status --short
    sha256sum c/glm53 "$COLI_USAGE"
}
fingerprint > "$run/before.txt"
printf 'Artifacts: %s\n' "$run"
python3 -B c/tests/glm53_kda_serve_lifecycle.py \
    --binary "$PWD/c/glm53" \
    --model /srv/models-fast/colibri/glm53-flash-i4 \
    --usage "$COLI_USAGE" --output "$run/stock" | tee "$run/console.log"
fingerprint > "$run/after.txt"
diff -u "$run/before.txt" "$run/after.txt"
printf 'PASS stock production SERVE validation. Artifacts: %s\n' "$run"
