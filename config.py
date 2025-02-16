class Config:
    def __init__(self):
        # Run settings
        self.RUN_NAME = "17-PPO-sde1x"
        self.RESUME_FROM = "logs/16-PPO-sde/latest-5000000-steps.zip"
        self.TOTAL_TIMESTEPS = 10_000_000
        self.LOG_DIR = "logs"
        
        # Model hyperparameters
        self.LEARNING_RATE = 3e-4
        self.GAMMA = 0.99
        
        # Evaluation and checkpoint settings
        self.EVAL_FREQ = 500_000
        self.EVAL_EPISODES = 1
        self.CHECK_FREQ = 50_000
        self.KEEP_N_MODELS = 1
