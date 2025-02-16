import os
import glob
import re
from pathlib import Path

def find_latest_model(checkpoint_dir="./checkpoints/"):
    # Convert to Path object
    checkpoint_path = Path(checkpoint_dir)
    
    # Check if directory exists
    if not checkpoint_path.exists():
        print("No checkpoints directory found!")
        return None
    
    # Check for final model first
    final_models = ["racing_model_final.zip", "best_model.zip"]
    for model_name in final_models:
        final_model = checkpoint_path / model_name
        if final_model.exists():
            return str(final_model)
    
    # Find all checkpoint files
    files = list(checkpoint_path.glob("*-steps-reward-*.zip"))
    if not files:
        print("No models found in checkpoints directory!")
        return None
        
    # Extract step numbers and find max
    steps = []
    for f in files:
        match = re.search(r'(\d+)-steps-reward-', f.name)
        if match:
            steps.append((int(match.group(1)), str(f)))
    
    if not steps:
        print("No valid model files found in checkpoints directory!")
        return None
        
    return max(steps)[1] if steps else None


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

def setup_run_dir(run_name: str, base_log_dir: str = "logs") -> Path:
    """
    Set up the run directory for logging using pathlib.
    
    Args:
        run_name: Name of the training run
        base_log_dir: Base directory for all logs
        
    Returns:
        Path: Path object pointing to the run directory
    """
    log_path = Path(base_log_dir)
    run_path = log_path / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path
