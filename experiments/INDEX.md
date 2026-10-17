# Experiment runs

One row per run in `experiments/`; held-out profit in INR/ha, mean ± std over 5 training seeds. Each run folder has its own run card.

| Study | Run | Algorithm | Overrides | Profit | Gap to threshold_tuned | Gap to oracle |
|---|---|---|---|---|---|---|
| 01_oracle_bound | [`baseline_v4_oracle`](01_oracle_bound/runs/baseline_v4_oracle/README.md) | ppo | – | 86,135 ± 179 | -1,096 ± 179 | -3,518 ± 179 |
| 02_improved_ppo | [`improved_ent0p01_ep10`](02_improved_ppo/runs/improved_ent0p01_ep10/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.01`, `train.epochs=10` | 85,864 ± 412 | -1,368 ± 412 | -3,789 ± 412 |
| 02_improved_ppo | [`improved_ent0p003_ep10`](02_improved_ppo/runs/improved_ent0p003_ep10/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=10` | 86,200 ± 542 | -1,032 ± 542 | -3,454 ± 542 |
| 02_improved_ppo | [`improved_ent0_ep10`](02_improved_ppo/runs/improved_ent0_ep10/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0`, `train.epochs=10` | 86,104 ± 554 | -1,128 ± 554 | -3,550 ± 554 |
| 02_improved_ppo | [`improved_ent0p01_ep4`](02_improved_ppo/runs/improved_ent0p01_ep4/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.01`, `train.epochs=4` | 85,820 ± 358 | -1,412 ± 358 | -3,834 ± 358 |
| 02_improved_ppo | [`improved_ent0p003_ep4`](02_improved_ppo/runs/improved_ent0p003_ep4/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=4` | 85,975 ± 386 | -1,257 ± 386 | -3,679 ± 386 |
| 02_improved_ppo | [`improved_ent0_ep4`](02_improved_ppo/runs/improved_ent0_ep4/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0`, `train.epochs=4` | 86,144 ± 271 | -1,088 ± 271 | -3,509 ± 271 |
| 03_action_structure | [`joint_nw`](03_action_structure/runs/joint_nw/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=10`, `train.head=joint_nw` | 86,250 ± 194 | -982 ± 194 | -3,403 ± 194 |
| 03_action_structure | [`fine_water`](03_action_structure/runs/fine_water/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=10`, `env.irrigation_levels=[0,25,50,75]` | 86,587 ± 286 | +196 ± 286 | -3,014 ± 286 |
| 04_other_algorithms | [`dqn_branching`](04_other_algorithms/runs/dqn_branching/README.md) | dqn | `train.algo=dqn`, `train.total_timesteps=1000000`, `train.lr=0.0005`, `train.gamma=1.0`, `train.hidden=128`, `train.batch_size=128` | 83,617 ± 1,741 | -3,615 ± 1,741 | -6,036 ± 1,741 |
| 04_other_algorithms | [`cem_threshold`](04_other_algorithms/runs/cem_threshold/README.md) | cem_threshold | `train.algo=cem_threshold` | 84,178 ± 192 | -3,054 ± 192 | -5,476 ± 192 |
| 05_pest_loss_prior | [`improved_best_pest035`](05_pest_loss_prior/runs/improved_best_pest035/README.md) | ppo | `train.variant=improved`, `train.ent_coef=0.003`, `train.epochs=10`, `pest_loss=0.35` | 86,410 ± 492 | +2,664 ± 492 | -4,151 ± 492 |
