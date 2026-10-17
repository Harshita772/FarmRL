#!/usr/bin/env bash
# Study 11, GRPO vs PPO. Waits for the follow-up chain (run_followup_studies.sh) to finish so
# the studies never compete for the machine or disturb each other's timings, then runs a
# timing probe and the three GRPO variants one after another (each with a pool of 4 processes,
# 1 torch thread each). Finished runs are skipped, so it can be started again after an
# interruption.
#
#   nohup setsid bash experiments/run_grpo_study.sh > experiments/grpo.log 2>&1 < /dev/null &
#
#   WAIT=0 bash experiments/run_grpo_study.sh        # start at once instead of waiting
set -u
PY=${PY:-/DATA/Aurindum/conda/envs/RL/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=$HERE/code
STUDY=11_grpo_vs_ppo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$HERE/.tmp
mkdir -p "$TMPDIR" "$HERE/$STUDY/logs"
stamp() { date '+%F %T'; }

if [ "${WAIT:-1}" = "1" ]; then
  echo "[$(stamp)] waiting for the follow-up chain to finish ($HERE/FOLLOWUP_DONE)"
  until [ -f "$HERE/FOLLOWUP_DONE" ]; do sleep 60; done
fi
cd "$CODE" || exit 1
COMMON="--multi-seed-train 5 --parallel-seeds 4 --torch-threads 1 --lean"
S1=$HERE/01_oracle_bound/runs/baseline_v4_oracle
TUNING=""; [ -f "$S1/baseline_tuning.json" ] && TUNING="--tuning-from $S1/baseline_tuning.json"
GRPO="--set train.variant=improved --set train.algo=grpo --set train.ent_coef=0.003 --set train.epochs=10 $TUNING"

run() {                      # run <name> <extra run_experiments args...>
  local name=$1; shift
  if [ -f "$HERE/$STUDY/runs/$name/metadata.json" ]; then echo "[$(stamp)] skip  $STUDY/$name (finished)"; return 0; fi
  echo "[$(stamp)] START $STUDY/$name"
  "$PY" scripts/run_experiments.py --runs-dir "$HERE/$STUDY/runs" --run-name "$name" $COMMON "$@" \
        > "$HERE/$STUDY/logs/$name.log" 2>&1
  echo "[$(stamp)] END   $STUDY/$name rc=$?"
}

echo "[$(stamp)] study 11 started"
if [ ! -f "$HERE/$STUDY/timing/time_probe.csv" ]; then
  echo "[$(stamp)] START timing probe"
  "$PY" scripts/time_probe.py --out "$HERE/$STUDY/timing" --steps 100000 > "$HERE/$STUDY/logs/time_probe.log" 2>&1
  echo "[$(stamp)] END   timing probe rc=$?"
fi
run grpo_g8           $GRPO --set train.group_size=8
run grpo_g16          $GRPO --set train.group_size=16
run grpo_g8_stepwise  $GRPO --set train.group_size=8 --set train.adv_mode=stepwise
"$PY" scripts/summarize_followup.py --experiments "$HERE" --study $STUDY >> "$HERE/$STUDY/logs/summary.log" 2>&1
echo "[$(stamp)] summary rc=$?"
echo "[$(stamp)] study 11 finished" | tee "$HERE/GRPO_DONE"
