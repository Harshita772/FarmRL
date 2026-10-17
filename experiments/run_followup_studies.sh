#!/usr/bin/env bash
# Follow-up studies 06-10, run one after another (each training run uses a pool of 4 processes
# with 1 torch thread each). Finished runs are skipped, so the script can be started again
# after an interruption; a failing step is logged and the chain carries on.
#
#   nohup setsid bash experiments/run_followup_studies.sh > experiments/followup.log 2>&1 < /dev/null &
#
# Needs the results of studies 01 and 02 (run_all_studies.sh). Everything it writes stays
# inside experiments/.
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

step() {                     # step <study> <name> <done-file> <command...>: a script that writes its own card
  local study=$1 name=$2 marker=$3; shift 3
  mkdir -p "$HERE/$study/logs"
  if [ -f "$marker" ]; then echo "[$(stamp)] skip  $study/$name (finished)"; return 0; fi
  echo "[$(stamp)] START $study/$name"
  "$@" > "$HERE/$study/logs/$name.log" 2>&1
  echo "[$(stamp)] END   $study/$name rc=$?"
}

summarize() {
  mkdir -p "$HERE/$1/logs"
  "$PY" scripts/summarize_followup.py --experiments "$HERE" --study "$1" >> "$HERE/$1/logs/summary.log" 2>&1
  echo "[$(stamp)] summary $1 rc=$?"
}

reuse() { if [ -f "$S1/baseline_tuning.json" ]; then echo "--tuning-from $S1/baseline_tuning.json"; fi; }

BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" 2>> "$HERE/followup_best.log")
S2BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" --path 2>/dev/null)
[ -z "$BEST" ] && BEST="--set train.ent_coef=0.003 --set train.epochs=10"
IMP="--set train.variant=improved $BEST"
FINE="--set baselines.fine_pests=true"
echo "[$(stamp)] follow-up chain started; improved PPO settings: $BEST; study-2 best run: $S2BEST"

# ---- study 6: pest-loss sweep (retune the rule, retrain PPO, recompute the oracle) -------
run 06_pest_loss_sweep pest_0p115 $IMP $FINE --oracle
run 06_pest_loss_sweep pest_0p2   $IMP $FINE --yield-param pest_loss=0.2 --oracle
run 06_pest_loss_sweep pest_0p3   $IMP $FINE --yield-param pest_loss=0.3 --oracle
run 06_pest_loss_sweep pest_0p35  $IMP $FINE --yield-param pest_loss=0.35 --oracle
run 06_pest_loss_sweep pest_0p5   $IMP $FINE --yield-param pest_loss=0.5 --oracle
summarize 06_pest_loss_sweep

# ---- study 10: what the oracle needs to know (cheap, replays study-2 models) -------------
step 10_oracle_information_breakdown breakdown "$HERE/10_oracle_information_breakdown/breakdown/README.md" \
  "$PY" scripts/oracle_breakdown.py --out "$HERE/10_oracle_information_breakdown/breakdown" \
        --base-run "$S1" --ppo-run "$S2BEST" --parallel 4

# ---- study 9: combined action settings ---------------------------------------------------
run 09_combined_action_settings fine4_joint   $IMP $FINE --set "env.irrigation_levels=[0,25,50,75]" --set train.head=joint_nw --oracle
run 09_combined_action_settings fine6_joint   $IMP $FINE --set "env.irrigation_levels=[0,15,30,45,60,75]" --set train.head=joint_nw --oracle
run 09_combined_action_settings fine6_factored $IMP $FINE --set "env.irrigation_levels=[0,15,30,45,60,75]" --oracle
summarize 09_combined_action_settings

# ---- study 7: forecast information -------------------------------------------------------
for sd in 0 10 25; do
  run 07_forecast_information "forecast_sd${sd}mm" $IMP --set env.forecast.enabled=true \
      --set env.forecast.noise_mm=$sd $(reuse) --oracle
done
summarize 07_forecast_information

# ---- study 8: transfer across regions and stress tests -----------------------------------
run 08_transfer_and_shift haryana_native $IMP $FINE --config configs/haryana.yaml --oracle
step 08_transfer_and_shift transfer "$HERE/08_transfer_and_shift/transfer/README.md" \
  "$PY" scripts/transfer_eval.py --out "$HERE/08_transfer_and_shift/transfer" --punjab-run "$S2BEST" \
        --punjab-rules "$S1" --haryana-run "$HERE/08_transfer_and_shift/runs/haryana_native" --parallel 4
summarize 08_transfer_and_shift 2>/dev/null || true

"$PY" scripts/summarize_followup.py --experiments "$HERE" --all >> "$HERE/followup_summary.log" 2>&1
echo "[$(stamp)] follow-up chain finished" | tee "$HERE/FOLLOWUP_DONE"
