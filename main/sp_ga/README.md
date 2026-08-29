# Surrogate-Prescreened Graph GA (SP-GA)

SP-GA keeps the official Graph GA crossover and mutation operators, but
generates a larger candidate pool and uses a random-forest surrogate to decide
which unseen candidates receive the limited oracle-call budget.

The acquisition score combines the surrogate mean, tree-to-tree uncertainty,
and fingerprint novelty. The surrogate is used only for prescreening. The
population fitness, PMO logs, and benchmark metrics always use the real oracle
scores.

## Run

```bash
python -c "from run import main; main('sp_ga', 'qed')" \
  --oracles qed --seed 0 --max_oracle_calls 1000 --wandb disabled
```

## Preliminary evaluation

An exploratory PMO-1K study evaluated five methods on QED, DRD2, GSK3B, JNK3,
and Celecoxib Rediscovery with ten paired seeds and 1,000 unique valid oracle
calls per run. SP-GA improved over Graph GA on Celecoxib Rediscovery and GSK3B
under paired bootstrap analysis. These are reduced-budget preliminary results,
not a complete official 23-task PMO evaluation.

Code, aggregate results, figures, limitations, and reproduction instructions:

- https://github.com/nkwork9999/pmo-experiments
- https://github.com/nkwork9999/pmo-experiments/blob/main/sub1-baseline/results/summary_all.csv

The next evaluation stage is the full 23-task, multi-seed grid at the standard
10,000-call budget.
