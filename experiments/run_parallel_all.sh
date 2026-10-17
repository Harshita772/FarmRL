#!/usr/bin/env bash
# Runs the follow-up studies 06-10 and the GRPO study 11 in parallel (run_followup_studies.sh
# and run_grpo_study.sh run the same work one run after another).
#
#   nohup setsid bash experiments/run_parallel_all.sh > experiments/parallel.log 2>&1 < /dev/null &
#
# Independent runs start together, at most MAXJ at a time (default 16, i.e. all of them; each
# run is a pool of 4 processes with 1 torch thread, about 60 processes on a 32-core /
# 64-thread machine, which keeps every hardware thread busy). Steps that depend on an earlier
# one stay in order:
#   * transfer_eval (study 8) needs the haryana_native run of the same study, so the two form
#     one job that runs them back to back;
#   * the wall-clock probe of study 11 runs alone once every run has finished, so that its
#     timings are not disturbed (the in-run timings of the GRPO runs are taken under load);
#   * the study cards and INDEX.md are written once all runs are done.
# Finished runs are skipped, so the script can be started again after an interruption.
set -u
PY=${PY:-/DATA/Aurindum/conda/envs/RL/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$HERE/.tmp
mkdir -p "$TMPDIR"
cd "$CODE" || exit 1
MAXJ=${MAXJ:-16}
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
  echo "[$(stamp)] END   $study/$name rc=$?"
}

step() {                     # step <study> <name> <done-file> <command...>
  local study=$1 name=$2 marker=$3; shift 3
  mkdir -p "$HERE/$study/logs"
  if [ -f "$marker" ]; then echo "[$(stamp)] skip  $study/$name (finished)"; return 0; fi
  echo "[$(stamp)] START $study/$name"
  "$@" > "$HERE/$study/logs/$name.log" 2>&1
  echo "[$(stamp)] END   $study/$name rc=$?"
}

launch() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done; "$@" & }

reuse() { if [ -f "$S1/baseline_tuning.json" ]; then echo "--tuning-from $S1/baseline_tuning.json"; fi; }
BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" 2>> "$HERE/parallel_best.log")
S2BEST=$("$PY" scripts/best_config.py "$HERE/02_improved_ppo/runs" --path 2>/dev/null)
[ -z "$BEST" ] && BEST="--set train.ent_coef=0.003 --set train.epochs=10"
IMP="--set train.variant=improved $BEST"
FINE="--set baselines.fine_pests=true"
GRPO="$IMP --set train.algo=grpo $(reuse)"
echo "[$(stamp)] parallel driver started (up to $MAXJ runs at once); settings: $BEST; study-2 best: $S2BEST"

# study 8: the transfer evaluation needs the Haryana run, so they run in order inside one job
haryana_then_transfer() {
  run 08_transfer_and_shift haryana_native $IMP $FINE --config configs/haryana.yaml --oracle
  step 08_transfer_and_shift transfer "$HERE/08_transfer_and_shift/transfer/README.md" \
    "$PY" scripts/transfer_eval.py --out "$HERE/08_transfer_and_shift/transfer" --punjab-run "$S2BEST" \
          --punjab-rules "$S1" --haryana-run "$HERE/08_transfer_and_shift/runs/haryana_native" --parallel 4
}
launch haryana_then_transfer

# study 10: replays the study-2 models, needs nothing new
launch step 10_oracle_information_breakdown breakdown "$HERE/10_oracle_information_breakdown/breakdown/README.md" \
  "$PY" scripts/oracle_breakdown.py --out "$HERE/10_oracle_information_breakdown/breakdown" \
        --base-run "$S1" --ppo-run "$S2BEST" --parallel 4

# study 6: pest-loss sweep (pest_0p115 is already finished and is skipped)
launch run 06_pest_loss_sweep pest_0p115 $IMP $FINE --oracle
for v in 0.2 0.3 0.35 0.5; do
  launch run 06_pest_loss_sweep "pest_$(echo $v | tr . p)" $IMP $FINE --yield-param pest_loss=$v --oracle
done

# study 9: combined action settings
launch run 09_combined_action_settings fine4_joint    $IMP $FINE --set "env.irrigation_levels=[0,25,50,75]" --set train.head=joint_nw --oracle
launch run 09_combined_action_settings fine6_joint    $IMP $FINE --set "env.irrigation_levels=[0,15,30,45,60,75]" --set train.head=joint_nw --oracle
launch run 09_combined_action_settings fine6_factored $IMP $FINE --set "env.irrigation_levels=[0,15,30,45,60,75]" --oracle

# study 7: forecast information (the tuned rules of study 1 are reused)
for sd in 0 10 25; do
  launch run 07_forecast_information "forecast_sd${sd}mm" $IMP --set env.forecast.enabled=true \
      --set env.forecast.noise_mm=$sd $(reuse) --oracle
done

# study 11: GRPO vs PPO
launch run 11_grpo_vs_ppo grpo_g8          $GRPO --set train.group_size=8
launch run 11_grpo_vs_ppo grpo_g16         $GRPO --set train.group_size=16
launch run 11_grpo_vs_ppo grpo_g8_stepwise $GRPO --set train.group_size=8 --set train.adv_mode=stepwise

wait                                       # every run, and the Haryana -> transfer job, has finished
echo "[$(stamp)] all runs finished"

# wall-clock probe of study 11, alone on the machine (up to 4 processes)
mkdir -p "$HERE/11_grpo_vs_ppo/logs"
if [ ! -f "$HERE/11_grpo_vs_ppo/timing/time_probe.csv" ]; then
  echo "[$(stamp)] START timing probe"
  "$PY" scripts/time_probe.py --out "$HERE/11_grpo_vs_ppo/timing" --steps 100000 > "$HERE/11_grpo_vs_ppo/logs/time_probe.log" 2>&1
  echo "[$(stamp)] END   timing probe rc=$?"
fi

for s in 06_pest_loss_sweep 09_combined_action_settings 07_forecast_information 08_transfer_and_shift 11_grpo_vs_ppo; do
  mkdir -p "$HERE/$s/logs"
  "$PY" scripts/summarize_followup.py --experiments "$HERE" --study "$s" >> "$HERE/$s/logs/summary.log" 2>&1
  echo "[$(stamp)] summary $s rc=$?"
done
"$PY" scripts/summarize_followup.py --experiments "$HERE" --all >> "$HERE/parallel_summary.log" 2>&1
echo "[$(stamp)] parallel driver finished" | tee "$HERE/PARALLEL_DONE"
