#!/usr/bin/env bash
# Study 12: the best action layout (4-level water grid + joint nitrogen x water head) together
# with a rain forecast at three noise levels. The three runs are independent and run in
# parallel (each a pool of 4 processes with 1 torch thread); the summary waits for all three.
#
#   nohup setsid bash experiments/run_layout_forecast.sh > experiments/layout_forecast.log 2>&1 < /dev/null &
#
# Needs studies 01, 02 and 09 (it reuses the rule tuning of 09's fine4_joint). Finished runs
# are skipped, so the script can be started again.
set -u
PY=${PY:-/DATA/Aurindum/conda/envs/RL/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/code
STUDY=12_layout_plus_forecast
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$HERE/.tmp
mkdir -p "$TMPDIR" "$HERE/$STUDY/logs"
cd "$CODE" || exit 1
COMMON="--multi-seed-train 5 --parallel-seeds 4 --torch-threads 1 --lean"
stamp() { date '+%F %T'; }

BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" 2>/dev/null)
[ -z "$BEST" ] && BEST="--set train.ent_coef=0.003 --set train.epochs=10"
TUNING=""; T9=$HERE/09_combined_action_settings/runs/fine4_joint/baseline_tuning.json
[ -f "$T9" ] && TUNING="--tuning-from $T9"
LAYOUT=(--set train.variant=improved --set baselines.fine_pests=true --set "env.irrigation_levels=[0,25,50,75]" --set train.head=joint_nw)

run() {                      # run <name> <forecast noise, mm>
  local name=$1 sd=$2
  if [ -f "$HERE/$STUDY/runs/$name/metadata.json" ]; then echo "[$(stamp)] skip  $STUDY/$name (finished)"; return 0; fi
  echo "[$(stamp)] START $STUDY/$name"
  "$PY" scripts/run_experiments.py --runs-dir "$HERE/$STUDY/runs" --run-name "$name" $COMMON "${LAYOUT[@]}" $BEST \
        --set env.forecast.enabled=true --set env.forecast.noise_mm=$sd $TUNING --oracle \
        > "$HERE/$STUDY/logs/$name.log" 2>&1
  echo "[$(stamp)] END   $STUDY/$name rc=$?"
}

echo "[$(stamp)] study 12 started"
run layout_fc0mm 0 &
run layout_fc10mm 10 &
run layout_fc25mm 25 &
wait
"$PY" scripts/summarize_followup.py --experiments "$HERE" --study $STUDY >> "$HERE/$STUDY/logs/summary.log" 2>&1
echo "[$(stamp)] summary rc=$?"
echo "[$(stamp)] study 12 finished" | tee "$HERE/LAYOUT_FORECAST_DONE"
