import os
import runpy


# =========================
# Unified plotting mode hyperparameter
# =========================

# Users can modify plot_mode here to trigger the plotting functionality
# of the corresponding original plot script.
# Available values and meanings are documented in the PLOT_MODES section
# at the bottom of this file and in utils/README.md.
plot_mode = "risk_threshold_metrics"


# Mapping of all plotting modes to script paths.
# NOTE: These scripts stay in the project root; this module only
# centralizes dispatching and provides a unified entry point.
PLOT_SCRIPTS = {
    # Source: plot_risk_threshold_metrics.py
    "risk_threshold_metrics": "plot_risk_threshold_metrics.py",

    # Source: plot_risk_confidence_mcf.py
    "risk_confidence_mcf": "plot_risk_confidence_mcf.py",

    # Source: plot_risk_confidence_mcf_linear.py
    "risk_confidence_mcf_linear": "plot_risk_confidence_mcf_linear.py",

    # Source: plot_risk_confidence_mcf_quadratic.py
    "risk_confidence_mcf_quadratic": "plot_risk_confidence_mcf_quadratic.py",

    # Source: plot_statistical_coefficients.py
    "statistical_coefficients": "plot_statistical_coefficients.py",

    # Source: plot_risk_confidence_mcf_isp.py
    "risk_confidence_mcf_isp": "plot_risk_confidence_mcf_isp.py",

    # Source: plot_risk_confidence_mcf_scf.py
    "risk_confidence_mcf_scf": "plot_risk_confidence_mcf_scf.py",

    # Source: plot_logistic_nonlinear_boundary_mcf_scf.py
    "logistic_nonlinear_boundary_mcf_scf": "plot_logistic_nonlinear_boundary_mcf_scf.py",

    # Source: plot_confusion_matrix_simple.py
    "confusion_matrix_simple": "plot_confusion_matrix_simple.py",

    # Source: plot_confusion_matrix_multi_threshold.py
    "confusion_matrix_multi_threshold": "plot_confusion_matrix_multi_threshold.py",

    # Source: plot_confusion_matrix_multi_disease.py
    "confusion_matrix_multi_disease": "plot_confusion_matrix_multi_disease.py",
}


def run_plot_mode(mode: str | None = None):
    """
    Run the plot script associated with the selected plot mode.

    :param mode: Plot mode string; if None, fall back to the global plot_mode.
    """
    if mode is None:
        mode = plot_mode

    if mode not in PLOT_SCRIPTS:
        raise ValueError(f"Unsupported plot mode: {mode}. Available modes: {list(PLOT_SCRIPTS.keys())}")

    filename = PLOT_SCRIPTS[mode]

    # Locate script relative to the project root
    utils_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(utils_dir)
    script_path = os.path.join(project_root, filename)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Plot script not found: {script_path}")

    # Use runpy to execute the script so that its __main__ entry point runs exactly as written
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    print(f"Current plot mode: {plot_mode}")
    print(f"Available modes: {list(PLOT_SCRIPTS.keys())}")
    run_plot_mode()



