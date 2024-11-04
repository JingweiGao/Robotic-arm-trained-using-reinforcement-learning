# Robotic-arm-trained-using-reinforcement-learning
Run Commands
    1. Installation
        pip install -r requirements.txt
    2. Run Experiment
        python play.py
    3. Retrain (This will overwrite saved models)
        python train.py

File Descriptions
algo/
    Reinforcement learning strategy algorithms
data/
    Data storage
logs/
    Stores training logs and trained models
result/
    Directory for storing result graphs
envs/env.py
    Reinforcement learning training environment
play.py
    Reinforcement learning experiment script
train.py
    Reinforcement learning training script
