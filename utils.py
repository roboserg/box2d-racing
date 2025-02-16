from pathlib import Path

def find_latest_model(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    
    def _find_models_in_dir(dir_path):
        # Check for final model first
        final_models = ["racing_model_final.zip", "best_model.zip"]
        for model_name in final_models:
            final_model = dir_path / model_name
            if final_model.exists():
                return str(final_model)
        
        # Find all checkpoint files
        files = list(dir_path.glob("*-steps-reward-*.zip"))
        if not files:
            return None
            
        # Extract step numbers and find max
        steps = []
        for f in files:
            parts = f.name.split("-steps-reward-")
            if len(parts) == 2 and parts[0].isdigit():
                steps.append((int(parts[0]), str(f)))
        
        return max(steps)[1] if steps else None

    # Check if directory exists
    if not checkpoint_path.exists():
        print("No checkpoints directory found!")
        return None
    
    # Try to find models in the specified directory
    model = _find_models_in_dir(checkpoint_path)
    if model:
        return model
        
    # If no models found, look for run directories (XX-name format)
    run_dirs = []
    for d in checkpoint_path.iterdir():
        if d.is_dir():
            parts = d.name.split("-", 1)
            if len(parts) == 2 and parts[0].isdigit():
                run_dirs.append((int(parts[0]), d))
    
    if not run_dirs:
        print("No run directories found!")
        return None
        
    # Get the latest run directory and search for models there
    latest_run_dir = max(run_dirs)[1]
    print(f"Searching in latest run directory: {latest_run_dir}")
    model = _find_models_in_dir(latest_run_dir)
    
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

def setup_run_dir(run_name: str, base_log_dir: str = "logs") -> Path:
    """
    Set up the run directory for logging using pathlib.
    If directory exists, increments the run number prefix.
    
    Args:
        run_name: Name of the training run (e.g. "04-test")
        base_log_dir: Base directory for all logs
        
    Returns:
        Path: Path object pointing to the run directory
    """
    log_path = Path(base_log_dir)
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
    return run_path
