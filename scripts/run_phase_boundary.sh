#!/bin/bash
# =============================================================================
# Phase Boundary Experiment — Fire and Forget
# =============================================================================
#
# Maps the critical learning rate η*(K) at which disambiguation becomes
# unreachable.  4 K values × 4-5 η values × 3 seeds = ~50 runs.
#
# Usage:
#   1. SSH into RunPod pod (1-4 GPUs recommended)
#   2. Clone repo, cd into it, activate venv / conda
#   3. Generate configs:
#        python scripts/generate_phase_boundary_configs.py
#   4. Launch:
#        nohup bash scripts/run_phase_boundary.sh > phase_boundary.log 2>&1 &
#   5. Disconnect.  Come back in ~24-48h.
#
# To monitor:
#   tail -f phase_boundary.log
#   cat outputs/phase_boundary_progress.txt
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

# ── Auto-detect parallelism ──
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    N_GPUS=1
fi
N_GPUS=${N_GPUS:-1}

PROGRESS_FILE="outputs/phase_boundary_progress.txt"
REUSE_MAP="configs/phase_boundary_reuse_map.txt"
RUN_LIST="configs/phase_boundary_run_list.txt"

mkdir -p outputs

echo "============================================================"
echo " Phase Boundary Experiment"
echo " Started:  $(date)"
echo " Repo:     ${REPO_DIR}"
echo " GPUs:     ${N_GPUS}"
echo "============================================================"
echo "STARTED $(date)" > "${PROGRESS_FILE}"
echo "GPUs: ${N_GPUS}"  >> "${PROGRESS_FILE}"

# ── Step 0: Generate configs if not already done ──
if [ ! -f "${RUN_LIST}" ]; then
    echo "Generating configs..."
    python scripts/generate_phase_boundary_configs.py
fi

# ── Step 1: Symlink/copy reused prior runs ──
if [ -f "${REUSE_MAP}" ]; then
    echo ""
    echo "Linking prior runs..."
    while IFS=$'\t' read -r NAME PRIOR; do
        DEST="outputs/${NAME}"
        SRC="outputs/${PRIOR}"
        if [ -d "${SRC}" ] && [ ! -d "${DEST}" ]; then
            echo "  ${NAME} ← ${PRIOR}"
            mkdir -p "${DEST}"
            # Copy training_history.json only (lightweight)
            cp "${SRC}/training_history.json" "${DEST}/" 2>/dev/null || true
            cp "${SRC}/config.yaml" "${DEST}/" 2>/dev/null || true
        fi
    done < "${REUSE_MAP}"
fi

# ── Step 2: Count runs ──
TOTAL=$(wc -l < "${RUN_LIST}" | tr -d ' ')
echo ""
echo "Runs to launch: ${TOTAL}"
echo "Max parallel:   ${N_GPUS}"
echo "Runs: ${TOTAL}" >> "${PROGRESS_FILE}"
echo "" >> "${PROGRESS_FILE}"

if [ "${TOTAL}" -eq 0 ]; then
    echo "Nothing to run — all experiments already complete."
    echo "Proceeding directly to analysis..."
    # Jump to analysis
    exec python scripts/analyze_phase_boundary.py
fi

# ── Step 3: Training loop with GPU scheduling ──
# We maintain an array of background PIDs, one per GPU slot.
# When a slot finishes, we launch the next run on that GPU.

echo ""
echo "============================================================"
echo " Launching training (${N_GPUS} GPUs)"
echo "============================================================"

COMPLETED=0
FAILED=0

declare -a SLOT_PID   # PID running in each GPU slot
declare -a SLOT_NAME  # Name of run in each GPU slot

for ((i=0; i<N_GPUS; i++)); do
    SLOT_PID[$i]=0
    SLOT_NAME[$i]=""
done

wait_for_slot() {
    # Block until at least one GPU slot is free.
    while true; do
        for ((i=0; i<N_GPUS; i++)); do
            pid=${SLOT_PID[$i]}
            if [ "$pid" -eq 0 ]; then
                echo "$i"
                return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished — check exit code
                wait "$pid" 2>/dev/null
                local rc=$?
                local name="${SLOT_NAME[$i]}"
                if [ $rc -eq 0 ]; then
                    echo "[$(date '+%H:%M:%S')] DONE  ${name}" >> "${PROGRESS_FILE}"
                    ((COMPLETED++)) || true
                else
                    echo "[$(date '+%H:%M:%S')] FAIL  ${name} (exit ${rc})" >> "${PROGRESS_FILE}"
                    echo "${rc}" > "outputs/${name}/FAILED"
                    ((FAILED++)) || true
                fi
                SLOT_PID[$i]=0
                SLOT_NAME[$i]=""
                echo "$i"
                return
            fi
        done
        sleep 15
    done
}

run_single() {
    local CONFIG_NAME=$1
    local GPU_ID=$2
    local OUT_DIR="outputs/${CONFIG_NAME}"
    local LOG_FILE="${OUT_DIR}/train.log"

    mkdir -p "${OUT_DIR}"

    # Skip if already trained
    if [ -f "${OUT_DIR}/training_history.json" ]; then
        echo "[SKIP] ${CONFIG_NAME} (already complete)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START ${CONFIG_NAME} on GPU ${GPU_ID}"
    echo "[$(date '+%H:%M:%S')] START ${CONFIG_NAME} gpu=${GPU_ID}" >> "${PROGRESS_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/train.py \
        --config-path ../configs/experiments \
        --config-name "${CONFIG_NAME}" \
        > "${LOG_FILE}" 2>&1

    return $?
}

# Read all run names into array
mapfile -t RUNS < "${RUN_LIST}"

for RUN_NAME in "${RUNS[@]}"; do
    # Skip empty lines
    [ -z "${RUN_NAME}" ] && continue

    # Skip if already complete
    if [ -f "outputs/${RUN_NAME}/training_history.json" ]; then
        echo "[SKIP] ${RUN_NAME}"
        ((COMPLETED++)) || true
        continue
    fi

    # Skip if config doesn't exist
    if [ ! -f "configs/experiments/${RUN_NAME}.yaml" ]; then
        echo "[ERROR] Missing config: configs/experiments/${RUN_NAME}.yaml"
        ((FAILED++)) || true
        continue
    fi

    # Wait for a free GPU slot
    FREE_SLOT=$(wait_for_slot)

    # Launch on that GPU
    run_single "${RUN_NAME}" "${FREE_SLOT}" &
    SLOT_PID[${FREE_SLOT}]=$!
    SLOT_NAME[${FREE_SLOT}]="${RUN_NAME}"

    sleep 3  # Brief pause to avoid filesystem contention
done

# ── Wait for all remaining slots to finish ──
echo ""
echo "All jobs launched. Waiting for stragglers..."
for ((i=0; i<N_GPUS; i++)); do
    pid=${SLOT_PID[$i]}
    if [ "$pid" -ne 0 ]; then
        wait "$pid" 2>/dev/null
        rc=$?
        name="${SLOT_NAME[$i]}"
        if [ $rc -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] DONE  ${name}" >> "${PROGRESS_FILE}"
            ((COMPLETED++)) || true
        else
            echo "[$(date '+%H:%M:%S')] FAIL  ${name} (exit ${rc})" >> "${PROGRESS_FILE}"
            mkdir -p "outputs/${name}"
            echo "${rc}" > "outputs/${name}/FAILED"
            ((FAILED++)) || true
        fi
    fi
done

echo "" >> "${PROGRESS_FILE}"
echo "ALL TRAINING COMPLETE: $(date)" >> "${PROGRESS_FILE}"
echo "Completed: ${COMPLETED}  Failed: ${FAILED}" >> "${PROGRESS_FILE}"

echo ""
echo "============================================================"
echo " Training complete: $(date)"
echo " Completed: ${COMPLETED}   Failed: ${FAILED}"
echo "============================================================"

# ── Step 4: Post-hoc analysis ──
echo ""
echo "Running analysis..."
python scripts/analyze_phase_boundary.py

echo ""
echo "============================================================"
echo " Phase Boundary Experiment DONE: $(date)"
echo " Results in: outputs/phase_boundary_summary.json"
echo " Figures in: outputs/paper_figures/fig_phase_boundary*.png"
echo "============================================================"
echo "EXPERIMENT COMPLETE: $(date)" >> "${PROGRESS_FILE}"
