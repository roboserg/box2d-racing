class Config:
    # Run settings
    RUN_NAME = "28-SAC-truncated"
    RESUME_FROM = "logs/27-SAC-vec-envs/1800000-steps-reward-53.6.zip"
    TOTAL_TIMESTEPS = 10_000_000
    LOG_DIR = "logs"
    NUM_ENVS = 10
    
    # Model hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    USE_SDE = False
    
    # Evaluation and checkpoint settings
    EVAL_FREQ = 500_000
    EVAL_EPISODES = 5
    CHECK_FREQ = 100_000
    KEEP_N_MODELS = 1
