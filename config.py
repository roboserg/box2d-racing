class Config:
    # Run settings
    RUN_NAME = "25-SAC-gamma99"
    RESUME_FROM = ""
    TOTAL_TIMESTEPS = 10_000_000
    LOG_DIR = "logs"
    
    # Model hyperparameters
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    USE_SDE = False
    
    # Evaluation and checkpoint settings
    EVAL_FREQ = 500_000
    EVAL_EPISODES = 5
    CHECK_FREQ = 50_000
    KEEP_N_MODELS = 1
