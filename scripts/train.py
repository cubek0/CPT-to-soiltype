import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig  # Import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install

from cpt_to_soiltype.schema_config import Config
from cpt_to_soiltype.train_eval_funcs import (evaluate_model, load_data,
                                              log_mlflow_metrics_and_model,
                                              train_predict)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """

    To rerun an experiment, use the following command:

    python <script_name>.py \
    --config-path <path_to_config> \
    --config-name <config_name> \
    +hydra=<hydra_config> \
    +overrides=<overrides_config>

    Example:
    python train.py \
    --config-path ./experiments/hydra_outputs/2024-11-12/14-25-36 \
    --config-name config \
    +hydra=@hydra.yaml \
    +overrides=@overrides.yaml
    """
    pcfg = Config(**OmegaConf.to_object(cfg))
    console = Console()
    console.print(pcfg)
    pcfg.mlflow.experiment_name = "train_test"

    # Load data
    console.print("[bold green]Loading training and testing data...[/bold green]")
    X_train, X_test, y_train, y_test = load_data(
        pcfg.dataset.path_model_ready_train,
        pcfg.dataset.path_model_ready_test,
        pcfg.experiment.label,
        pcfg.experiment.features,
    )

    # Train model and predict output
    console.print("[bold green]Train and predict...[/bold green]")
    y_pred = train_predict(
        pcfg.model.name,
        pcfg.model.params,
        X_train,
        X_test,
        y_train,
        y_test,
        pcfg.experiment.undersample_level,
        pcfg.experiment.oversample_level,
        pcfg.experiment.save_model,
    )

    # Evaluate model
    console.print("[bold green]Evaluating model...[/bold green]")
    metrics, artifacts = evaluate_model(
        y_test, y_pred, pcfg.experiment.soil_classification
    )

    # log results and config to mlflow
    console.print("[bold green]Logging data to MLflow...[/bold green]")
    hydra_cfg_dir = Path(HydraConfig.get().run.dir) / ".hydra"
    log_mlflow_metrics_and_model(
        pcfg.mlflow.path,
        pcfg.mlflow.experiment_name,
        metrics,
        artifacts,
        pcfg.model.name,
        pcfg.model.params,
        pcfg.experiment.undersample_level,
        pcfg.experiment.oversample_level,
        hydra_cfg_dir,
    )


if __name__ == "__main__":
    install()
    os.environ["HYDRA_FULL_ERROR"] = "1"
    console = Console()
    main()
