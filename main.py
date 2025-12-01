import argparse

from config.config import AVAILABLE_MODES, MODE_CONFIGS, run_mode
from dataloader.data_loader import (
    create_train_test,
    generate_common_testing_csv,
    generate_cross_data_pickle,
    generate_cross_training_csv,
)
from models.model import (
    compute_statistical_coefficients,
    generate_threshold_metrics,
    prepare_confidence_artifacts,
    print_threshold_evaluation,
    reevaluate_models,
    train_ensemble_models,
)
from utils.utils import logger


TASKS = {
    "af_data_prep": create_train_test,
    "generate_cross_training_csv": generate_cross_training_csv,
    "generate_common_testing_csv": generate_common_testing_csv,
    "generate_cross_data_pickle": generate_cross_data_pickle,
    "train_models": train_ensemble_models,
    "reevaluate_models": reevaluate_models,
    "generate_threshold_metrics": generate_threshold_metrics,
    "print_threshold_evaluation": print_threshold_evaluation,
    "compute_statistical_coefficients": compute_statistical_coefficients,
    "prepare_confidence_artifacts": prepare_confidence_artifacts,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified entry point")
    parser.add_argument("--run_mode", choices=AVAILABLE_MODES, default=run_mode, help="specify run mode")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.run_mode or run_mode
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported run mode: {mode}, available modes: {AVAILABLE_MODES}")
    cfg = MODE_CONFIGS[mode]
    if mode not in TASKS:
        raise ValueError(f"Run mode {mode} has not been registered with a task yet")
    logger.info("Current run mode: %s", mode)
    TASKS[mode](cfg)


if __name__ == "__main__":
    main()

