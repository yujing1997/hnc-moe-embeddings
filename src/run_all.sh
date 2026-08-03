#!/bin/bash
# run_all.sh — Orchestrate embedding generation + survival comparison
#
# Usage:
#   bash run_all.sh              # Full pipeline (all models, 1000 bootstrap)
#   bash run_all.sh --quick      # Quick test (longformer only, 50 bootstrap)
#   bash run_all.sh --embed-only # Only generate embeddings
#   bash run_all.sh --compare-only # Only run comparison (assumes embeddings exist)
#
# Environment: conda activate llm_extr

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../outputs"
RESULTS_DIR="${SCRIPT_DIR}/../results"

# Defaults
N_BOOTSTRAP=1000
MODELS="all"
OUTCOME="all"
RUN_EMBED=true
RUN_COMPARE=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            N_BOOTSTRAP=50
            MODELS="longformer"
            OUTCOME="DFS"
            echo ">>> QUICK MODE: longformer only, DFS, 50 bootstrap"
            ;;
        --embed-only)
            RUN_COMPARE=false
            ;;
        --compare-only)
            RUN_EMBED=false
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash run_all.sh [--quick] [--embed-only] [--compare-only]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Embeddings vs. Structured Extraction Comparison Pipeline"
echo "============================================================"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Bootstrap:   ${N_BOOTSTRAP}"
echo "Models:      ${MODELS}"
echo "Outcome:     ${OUTCOME}"
echo "============================================================"

# Step 1: Generate embeddings
if [ "$RUN_EMBED" = true ]; then
    echo ""
    echo ">>> Step 1: Generating embeddings..."
    echo ""
    python "${SCRIPT_DIR}/01_generate_embeddings.py" \
        --model "${MODELS}" \
        --output_dir "${OUTPUT_DIR}"
    echo ""
    echo ">>> Embeddings complete."
fi

# Step 2: Survival comparison
if [ "$RUN_COMPARE" = true ]; then
    echo ""
    echo ">>> Step 2: Running survival comparison..."
    echo ""
    python "${SCRIPT_DIR}/02_survival_comparison.py" \
        --n_bootstrap "${N_BOOTSTRAP}" \
        --outcome "${OUTCOME}" \
        --embeddings_dir "${OUTPUT_DIR}" \
        --output_dir "${RESULTS_DIR}"
    echo ""
    echo ">>> Comparison complete."
fi

echo ""
echo "============================================================"
echo "Pipeline finished. Results in: ${RESULTS_DIR}"
echo "============================================================"
