#!/usr/bin/env bash
#
# Submit multi-node Night Science training to a Ray cluster.
#
#   set -a; . ./.env; set +a      # load your configuration
#   ./launch.sh                   # uses configs/night_science_8b.yaml
#   ./launch.sh --config-name my_config
#
# Single node? Skip Ray entirely and run:
#   bash examples/sglang_multiturn/day_night/run_qwen2.5-3b_instruct_proposal_gen_multiturn.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${OUTPUTS:?OUTPUTS is not set — copy .env.example to .env and fill it in}"
: "${CREATIVE_AZURE_ENDPOINTS:?CREATIVE_AZURE_ENDPOINTS is not set — the reward needs an LLM judge (see .env.example)}"

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
CONFIG_NAME="night_science_8b"
if [ "${1:-}" = "--config-name" ]; then CONFIG_NAME="$2"; shift 2; fi

export PROPOSAL_GEN_RUN_TIMESTAMP="${PROPOSAL_GEN_RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

# Render the runtime env template, substituting only our two placeholders.
RENDERED="$(mktemp /tmp/night_science_runtime_env.XXXXXX.yaml)"
trap 'rm -f "$RENDERED"' EXIT
OUTPUTS="$OUTPUTS" PROPOSAL_GEN_RUN_TIMESTAMP="$PROPOSAL_GEN_RUN_TIMESTAMP" \
python3 -c '
import os, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
for key in ("OUTPUTS", "PROPOSAL_GEN_RUN_TIMESTAMP"):
    text = text.replace("$" + key, os.environ[key])
open(dst, "w", encoding="utf-8").write(text)
' setup/runtime_env.yaml "$RENDERED"

echo "Submitting to $RAY_ADDRESS  (config: $CONFIG_NAME, run: $PROPOSAL_GEN_RUN_TIMESTAMP)"
ray job submit --address="$RAY_ADDRESS" --runtime-env="$RENDERED" -- \
    python3 -m verl.trainer.main_ppo --config-dir=configs/ --config-name="$CONFIG_NAME" "$@"
