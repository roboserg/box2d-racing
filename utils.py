import os
import glob
import re

def find_latest_model(checkpoint_dir="./checkpoints/"):
    # Check for final model first
    final_model = os.path.join(checkpoint_dir, "racing_model_final.zip")
    if os.path.exists(final_model):
        return final_model
    
    # Find all checkpoint files
    files = glob.glob(os.path.join(checkpoint_dir, "racing_model_*.zip"))
    if not files:
        return None
        
    # Extract step numbers and find max
    steps = []
    for f in files:
        match = re.search(r'racing_model_(\d+)_steps\.zip', f)
        if match:
            steps.append((int(match.group(1)), f))
    
    return max(steps)[1] if steps else None
