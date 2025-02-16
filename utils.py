import json
import yaml
from pathlib import Path
from dataclasses import asdict
from typing import Literal

def _find_models_in_dir(dir_path: Path, mode: Literal["best", "last"]) -> str | None:
    """
    Search for model files in a directory, prioritizing final models over checkpoints.
    
    The function first looks for final model files ('racing_model_final.zip' or 'best_model.zip').
    If none are found, it searches for checkpoint files in the format '*-steps-reward-*.zip'
    and returns the one with the highest step count.
    
    Args:
        dir_path: Directory path to search for model files
        mode: Either "best" for best performing model or "last" for most recent checkpoint
        
    Returns:
        str: Path to the found model file, or None if no models are found
    """
    if mode == "best":
        # Search for best model files first
        final_models = ["racing_model_final.zip", "best_model.zip"]
        for model_name in final_models:
            final_model = dir_path / model_name
            if final_model.exists():
                return str(final_model)
        
        # Then search for checkpoint files with rewards
        files = list(dir_path.glob("*-steps-reward-*.zip"))
        if not files:
            return None
            
        steps = []
        for f in files:
            parts = f.name.split("-steps-reward-")
            if len(parts) == 2 and parts[0].isdigit():
                steps.append((int(parts[0]), str(f)))
        
        return max(steps)[1] if steps else None
        
    elif mode == "last":
        # Search for latest checkpoint files
        files = list(dir_path.glob("latest-*-steps.zip"))
        if not files:
            return None
            
        steps = []
        for f in files:
            parts = f.name.split("-")
            if len(parts) == 3 and parts[1].isdigit():
                steps.append((int(parts[1]), str(f)))
        
        return max(steps)[1] if steps else None
    
    return None


def find_latest_model(checkpoint_dir: str | Path, mode: Literal["best", "last"] = "best") -> str | None:
    """
    Find the most recent model file in the checkpoint directory hierarchy.
    
    The function searches in the following order:
    1. Direct search in the specified directory
    2. If no models found, search in run subdirectories (format: XX-name)
    3. In run subdirectories, select the one with highest prefix number
    
    Args:
        checkpoint_dir: Base directory to search for model files
        
    Returns:
        str: Path to the latest model file, or None if no models are found
        
    Example:
        >>> find_latest_model("logs/training")
        'logs/training/07-test/best_model.zip'
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print("No checkpoints directory found!")
        return None
    
    model = _find_models_in_dir(checkpoint_path, mode)
    if model:
        return model
        
    run_dirs = []
    for d in checkpoint_path.iterdir():
        if d.is_dir():
            parts = d.name.split("-", 1)
            if len(parts) == 2 and parts[0].isdigit():
                run_dirs.append((int(parts[0]), d))
    
    if not run_dirs:
        print("No run directories found!")
        return None
        
    latest_run_dir = max(run_dirs)[1]
    print(f"Searching in latest run directory: {latest_run_dir}")
    model = _find_models_in_dir(latest_run_dir, mode)
    
    if not model:
        print("No models found in the latest run directory!")
    
    return model


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: Function that computes the current learning rate.
    """
    def schedule(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0 (end).

        :param progress_remaining: (float) Current progress remaining (from 1 to 0).
        :return: (float) Current learning rate.
        """
        return progress_remaining * initial_value

    return schedule

def setup_run_dir(config) -> Path:
    """
    Set up the run directory for logging using pathlib.
    If directory exists, increments the run number prefix.
    Dumps config as JSON and YAML in the run directory.
    
    Args:
        config: Config class containing run configuration (must have RUN_NAME attribute)
        base_log_dir: Base directory for all logs
        
    Returns:
        Path: Path object pointing to the run directory
    """
    run_name = config.RUN_NAME
    log_path = Path(config.LOG_DIR)
    run_path = log_path / run_name
    
    if run_path.exists():
        parts = run_name.split("-", 1)
        if len(parts) == 2 and parts[0].isdigit():
            num, name = parts
            current = int(num)
            while (log_path / f"{current:02d}-{name}").exists():
                current += 1
            run_name = f"{current:02d}-{name}"
            run_path = log_path / run_name
            print(f"Run directory already exists. Using new name: {run_name}")
    
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    with open(run_path / "config.yaml", "w") as f:
        yaml.safe_dump(vars(config), f, default_flow_style=False)
    
    return run_path
