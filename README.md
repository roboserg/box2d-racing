# Box2D Racing Environment

This repository contains a custom racing environment built using the Box2D physics engine and the Gymnasium library. The environment is designed for reinforcement learning experiments, particularly using the Soft Actor-Critic (SAC) algorithm from the Stable Baselines3 library.

## Features

- **Custom Racing Environment**: A racing track is generated using Box2D, with customizable parameters for track size and complexity.
- **Reinforcement Learning**: The environment is compatible with Gymnasium, allowing for easy integration with reinforcement learning algorithms.
- **Soft Actor-Critic (SAC) Training**: The repository includes scripts to train a model using the SAC algorithm.
- **Evaluation and Visualization**: Tools are provided to evaluate trained models and visualize their performance in the environment.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd box2d-racing
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Training**: Run `train.py` to start training a model.
- **Evaluation**: Use `eval.py` to evaluate a trained model.
- **Manual Play**: Execute `play.py` to manually control the car in the environment.