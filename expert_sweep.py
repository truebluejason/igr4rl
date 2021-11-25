import os
import wandb

from expert_train import main

if __name__ == "__main__":
    sweep_id = os.environ["SWEEP_ID"]
    wandb.agent(entity='jasonyoo', project='igr4rl', sweep_id=sweep_id, function=main)
