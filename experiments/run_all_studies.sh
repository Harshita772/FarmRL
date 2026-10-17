#!/usr/bin/env bash
# Runs every study in order, one run after another (each run trains its 5 seeds in a pool of
# 4 processes with 1 torch thread each). Finished runs are skipped, so the script can be
# started again after an interruption. A failing step is logged and the chain carries on.
#
#   nohup setsid bash experiments/run_all_studies.sh > experiments/overnight.log 2>&1 &
#
# Everything it writes stays inside experiments/: runs under <study>/runs, logs under
# <study>/logs, the study cards and summary.png next to them.
set -u
PY=${PY:-/DATA/Aurindum/conda/envs/RL/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$HERE/.tmp
mkdir -p "$TMPDIR"
cd "$CODE" || exit 1
COMMON="--multi-seed-train 5 --parallel-seeds 4 --torch-threads 1 --lean"
S1=$HERE/01_oracle_bound/runs/baseline_v4_oracle

stamp() { date '+%F %T'; }

run() {                      # run <study> <run name> <extra run_experiments args...>
  local study=$1 name=$2; shift 2
  mkdir -p "$HERE/$study/logs"
  if [ -f "$HERE/$study/runs/$name/metadata.json" ]; then echo "[$(stamp)] skip  $study/$name (finished)"; return 0; fi
  echo "[$(stamp)] START $study/$name"
  "$PY" scripts/run_experiments.py --runs-dir "$HERE/$study/runs" --run-name "$name" $COMMON "$@" \
        > "$HERE/$study/logs/$name.log" 2>&1
  local rc=$?
  echo "[$(stamp)] END   $study/$name rc=$rc"
  return $rc
}

summarize() {                # summarize <study>
  mkdir -p "$HERE/$1/logs"
  "$PY" scripts/summarize_study.py --experiments "$HERE" --study "$1" >> "$HERE/$1/logs/summary.log" 2>&1
  echo "[$(stamp)] summary $1 rc=$?"
}

# reuse the rule tuning of study 1 when the environment is the same
reuse() { if [ -f "$S1/baseline_tuning.json" ]; then echo "--tuning-from $S1/baseline_tuning.json"; fi; }

echo "[$(stamp)] chain started in $HERE"

# ---- study 1: hindsight oracle (also reproduces v4: original PPO, 200k steps) ----------
run 01_oracle_bound baseline_v4_oracle --oracle
summarize 01_oracle_bound

# ---- study 2: improved PPO bundle + (ent_coef x epochs) sweep, default setting first ---
for ep in 10 4; do
  for ent in 0.01 0.003 0; do
    tag=$(echo "$ent" | tr . p)
    run 02_improved_ppo "improved_ent${tag}_ep${ep}" --set train.variant=improved \
        --set train.ent_coef=$ent --set train.epochs=$ep $(reuse)
  done
done
summarize 02_improved_ppo

# ---- best study-2 configuration feeds studies 3 and 5 -----------------------------------
BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" 2>> "$HERE/overnight_best.log")
echo "[$(stamp)] best study-2 settings: $BEST"
if [ -z "$BEST" ]; then BEST="--set train.ent_coef=0.01 --set train.epochs=10"; echo "using the default settings instead"; fi

# ---- study 3: action structure ----------------------------------------------------------
run 03_action_structure joint_nw --set train.variant=improved $BEST --set train.head=joint_nw $(reuse)
run 03_action_structure fine_water --set train.variant=improved $BEST \
    --set "env.irrigation_levels=[0,25,50,75]" --oracle
summarize 03_action_structure

# ---- study 4: other algorithms ----------------------------------------------------------
run 04_other_algorithms dqn_branching --set train.algo=dqn --set train.total_timesteps=1000000 \
    --set train.lr=0.0005 --set train.gamma=1.0 --set train.hidden=128 --set train.batch_size=128 $(reuse)
run 04_other_algorithms cem_threshold --set train.algo=cem_threshold $(reuse)
summarize 04_other_algorithms

# ---- study 5: old pest-loss prior -------------------------------------------------------
run 05_pest_loss_prior improved_best_pest035 --set train.variant=improved $BEST \
    --yield-param pest_loss=0.35 --oracle
summarize 05_pest_loss_prior

"$PY" scripts/summarize_study.py --experiments "$HERE" --all >> "$HERE/overnight_summary.log" 2>&1
echo "[$(stamp)] summary of all studies rc=$?"
echo "[$(stamp)] chain finished" | tee "$HERE/DONE"
