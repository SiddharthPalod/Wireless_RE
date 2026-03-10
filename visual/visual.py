import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("training_rewards.npy")
losses = np.load("training_losses.npy")
actions = np.load("training_actions.npy")

steps = np.arange(len(rewards))

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(14,10))

# Reward plot
plt.subplot(3,1,1)
plt.plot(steps, rewards, alpha=0.3, label="reward")
plt.plot(moving_average(rewards), label="reward moving avg")
plt.title("Reward vs Training Step")
plt.ylabel("Reward")
plt.legend()

# Loss plot
plt.subplot(3,1,2)
plt.plot(steps, losses)
plt.title("Loss vs Training Step")
plt.ylabel("Loss")

# Action plot
plt.subplot(3,1,3)
plt.scatter(steps, actions, s=5)
plt.title("Action vs Training Step")
plt.ylabel("RIS partition action")
plt.xlabel("Training step")

plt.tight_layout()
plt.show()

# Action histogram over training
plt.figure()
plt.hist(actions, bins=30)
plt.xlabel("RIS partition")
plt.ylabel("Frequency")
plt.title("Action distribution")
plt.show()