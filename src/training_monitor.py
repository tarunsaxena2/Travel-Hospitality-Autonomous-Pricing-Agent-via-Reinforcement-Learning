"""
training_monitor.py

Utility for tracking and visualizing DQN training stability: 
loss curves and episodic reward curves across training steps.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(loss_history, reward_history, save_path=None):
    """
    Plots training loss and episodic reward curves side by side 
    to help identify signs of divergence during DQN training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_history)
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")

    axes[1].plot(reward_history)
    axes[1].set_title("Episodic Reward Curve")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Reward")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def detect_divergence(loss_history, window=50, threshold=2.0):
    """
    Simple heuristic: flags divergence if recent average loss is 
    significantly higher than an earlier window's average loss.
    """
    if len(loss_history) < window * 2:
        return False

    early_avg = np.mean(loss_history[:window])
    recent_avg = np.mean(loss_history[-window:])

    if early_avg == 0:
        return False

    return recent_avg > early_avg * threshold