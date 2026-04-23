# Plot-generation scripts

These scripts regenerate the five figures embedded in `report.pdf`.
Each is standalone: it loads `data/features.npz`, trains the relevant
models from `src/methods/`, and writes a PNG into `plots/`.

| Script                       | Output                              | Referenced as |
|------------------------------|-------------------------------------|---------------|
| `all_sweeps_plot.py`         | `plots/all_sweeps.png`              | Fig. 1        |
| `linreg_plot.py`             | `plots/linreg_summary.png`          | Fig. 2        |
| `logreg_plot.py`             | `plots/logreg_summary.png`          | Fig. 3        |
| `knn_plot.py`                | `plots/knn_summary.png`             | Fig. 4        |
| `confusion_matrices_plot.py` | `plots/confusion_matrices.png`      | Fig. 5        |

## How to run

All scripts must be invoked **from the project root** so their imports
(`from src.methods...`) and data path (`data/features.npz`) resolve:

```bash
python3 scripts/all_sweeps_plot.py
python3 scripts/linreg_plot.py
python3 scripts/logreg_plot.py
python3 scripts/knn_plot.py
python3 scripts/confusion_matrices_plot.py
```

Requires `data/features.npz` to be present in `data/`.
