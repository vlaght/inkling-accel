#!/bin/sh
# Inkling 2 Mini with DSpark speculative decoding on llama.cpp (CUDA, the patched tree): OpenAI-compatible API with tool calling.
# Settings come from the environment (docs/reproduce.md); the launcher adds nothing else.
#   ROOT, BUILD, MODEL, DRAFT, CUDA   as for the measuring scripts
#   HOST (default 0.0.0.0)  PORT (default 8080)  API_KEY_FILE (adds --api-key when set)
#   CTX (default 65536)  UB (default 32): CTX=131072 UB=8 is the long-context mode (the compute buffer grows with CTX x UB)
#   NSPEC (default 2): drafted tokens per step
# About 15 GB of VRAM at the defaults: stop other GPU users first.
ROOT=${ROOT:-$HOME/llama.cpp}; BUILD=${BUILD:-build-cuda}; CUDA=${CUDA:-/usr/local/cuda}
MODEL=${MODEL:-$HOME/models/inkling-2-mini/Inkling-2-Mini-UD-IQ3_XXS.gguf}
DRAFT=${DRAFT:-$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf}
export LD_LIBRARY_PATH=$ROOT/$BUILD/bin:$CUDA/lib64
KEY=""; [ -n "$API_KEY_FILE" ] && KEY="--api-key $(cat "$API_KEY_FILE")"
exec "$ROOT/$BUILD/bin/llama-server" -m "$MODEL" -ngl 99 -c "${CTX:-65536}" -ub "${UB:-32}" -b "${UB:-32}" \
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 --model-draft "$DRAFT" --spec-type draft-dspark --spec-draft-ngl 99 --spec-draft-n-max "${NSPEC:-2}" \
  --jinja --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" $KEY
