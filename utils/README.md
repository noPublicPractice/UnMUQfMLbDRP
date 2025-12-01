## Plot Utility Overview (`utils/plot_utils.py`)

This module centralizes every legacy `*-plot-*.py` visualization via a single **`plot_mode` hyperparameter + dispatcher**. Each plot script now has an English filename in the project root, but its original logic (including `if __name__ == "__main__"` blocks) remains untouched.

### 1. Unified entry file

- **Location**: `utils/plot_utils.py`
- **Key APIs**:
  - `plot_mode`: string hyperparameter that selects which visualization to run.
  - `run_plot_mode(mode: str | None = None)`: dispatch to the plot associated with `mode` or fall back to the global `plot_mode`.
  - Running `python utils/plot_utils.py` from the project root will execute the currently selected plot.

### 2. Plot modes and script mapping

| `plot_mode` | Script | Purpose |
|-------------|--------|---------|
| `risk_threshold_metrics` | `plot_risk_threshold_metrics.py` | ACC/PPV/TPR/F1 curves vs. binary risk thresholds |
| `risk_confidence_mcf` | `plot_risk_confidence_mcf.py` | Risk vs. confidence scatter plot (MCF, control group) |
| `risk_confidence_mcf_linear` | `plot_risk_confidence_mcf_linear.py` | Scatter plot + linear decision boundary evaluation |
| `risk_confidence_mcf_quadratic` | `plot_risk_confidence_mcf_quadratic.py` | Scatter plot + quadratic decision boundary evaluation |
| `statistical_coefficients` | `plot_statistical_coefficients.py` | Statistical coefficients / interval proportions (ISP / SCF) |
| `risk_confidence_mcf_isp` | `plot_risk_confidence_mcf_isp.py` | Risk vs. confidence scatter plot for MCF×ISP |
| `risk_confidence_mcf_scf` | `plot_risk_confidence_mcf_scf.py` | Risk vs. confidence scatter plot for MCF×SCF (multiple legend layouts) |
| `logistic_nonlinear_boundary_mcf_scf` | `plot_logistic_nonlinear_boundary_mcf_scf.py` | Logistic-regression nonlinear boundary (MCF×SCF experimental group) |
| `confusion_matrix_simple` | `plot_confusion_matrix_simple.py` | Single 2×2 confusion matrix heatmap |
| `confusion_matrix_multi_threshold` | `plot_confusion_matrix_multi_threshold.py` | Multi-threshold / multi-confidence confusion matrices |
| `confusion_matrix_multi_disease` | `plot_confusion_matrix_multi_disease.py` | Multi-disease, multi-confidence confusion matrices with TP/FN/FP/TN views |

### 3. Usage options

**Option A: tweak `plot_mode` and run**
1. Open `utils/plot_utils.py`.
2. Set `plot_mode = "<desired_mode>"`.
3. From the project root execute `python utils/plot_utils.py`.
4. The dispatcher loads the selected script and runs it exactly as authored.

**Option B: call from other modules**
```python
from utils.plot_utils import run_plot_mode

run_plot_mode()                       # Uses the global plot_mode
run_plot_mode("risk_confidence_mcf_scf")  # Explicit override
```

### 4. Design notes

- **Single source of truth**: Each plot script is still responsible for its own logic; the dispatcher simply routes execution.
- **`plot_mode` + dictionary**: Replaces a long `if/elif` chain with a maintainable mapping.
- **Extending the catalog**:
  1. Create a new plot script (e.g., `plot_new_insight.py`).
  2. Add `mode → filename` to `PLOT_SCRIPTS`.
  3. Document the mode in this README.


