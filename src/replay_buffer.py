import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer for DQN training.
    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)

    # Push some dummy transitions
    for _ in range(50):
        state = np.random.uniform(low=[0, 0], high=[100, 30], size=2)
        action = np.random.randint(0, 10)
        reward = np.random.uniform(0, 50)
        next_state = np.random.uniform(low=[0, 0], high=[100, 30], size=2)
        done = np.random.choice([True, False])
        buffer.push(state, action, reward, next_state, done)

    print("Buffer size:", len(buffer))

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=8)
    print("Sampled batch shapes:")
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  rewards:", rewards.shape)
    print("  next_states:", next_states.shape)
    print("  dones:", dones.shape)