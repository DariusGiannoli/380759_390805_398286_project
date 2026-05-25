# Milestone 2 Reproducibility

Place `features.npz` in `data/features.npz`, then run from this folder:

```bash
bash reproduce.sh
```

The script regenerates `figures/summary.json`, all report figures, `report.pdf`,
and `ms2_report.pdf`. If your NumPy environment is not the default `python3`,
set `PYTHON` explicitly:

```bash
PYTHON=/path/to/python bash reproduce.sh
```

Optional implementation checks:

```bash
python3 test_ms2.py
python3 gradient_checks.py
```

## Final Commands

MLP classification:

```bash
python3 main.py --data_path data/features.npz --method mlp --task classification --hidden_dims 64,32 --activation relu --lr 0.003 --epochs 40 --batch_size 32 --beta 0.9 --loss ce --test
```

MLP regression:

```bash
python3 main.py --data_path data/features.npz --method mlp --task regression --hidden_dims 64,32 --activation relu --lr 0.0003 --epochs 40 --batch_size 32 --beta 0.9 --test
```

K-Means classification:

```bash
python3 main.py --data_path data/features.npz --method kmeans --task classification --K 50 --max_iters 200 --kmeans_init kmeans++ --kmeans_restarts 10 --test
```
