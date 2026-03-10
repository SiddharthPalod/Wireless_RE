import random
from collections import deque

import torch

from dqn import DQN
from simple_ris_env import SimpleRISEnv


def run():
    env = SimpleRISEnv()

    state_dim = 2  # [dis_user2, partition]
    n_actions = env.N + 1  # N1 in [0, N]

    model = DQN(state_dim=state_dim, n_actions=n_actions)
    target_model = DQN(state_dim=state_dim, n_actions=n_actions)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    # simple replay buffer (experience replay)
    buffer_capacity = 10_000
    batch_size = 32
    replay_buffer = deque(maxlen=buffer_capacity)
    target_update_freq = 20  # steps between target network updates

    def preprocess(state):
        # normalize to roughly [0,1] ranges for stability
        dis_user2 = float(state[0]) / 20.0
        partition = float(state[1]) / float(env.N)
        return torch.FloatTensor([dis_user2, partition])

    state = env.reset()
    state = preprocess(state)
    episode = 0

    last_loss = torch.tensor(0.0)
    global_step = 0

    for step in range(200):
        # get Q-values for current state
        q_values = model(state)

        # epsilon-greedy action selection (prevents premature convergence)
        if torch.rand(()) < epsilon:
            action = int(torch.randint(low=0, high=n_actions, size=(1,)).item())
        else:
            action = int(torch.argmax(q_values).item())

        # environment step
        state_next, reward, done = env.step(action)

        state_next_tensor = preprocess(state_next)

        # store transition in replay buffer
        replay_buffer.append(
            (state.detach(), action, float(reward), state_next_tensor.detach(), done)
        )

        state = state_next_tensor

        # train from replay buffer if enough samples collected
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

            states_b = torch.stack(states_b)  # [B, state_dim]
            next_states_b = torch.stack(next_states_b)  # [B, state_dim]
            actions_b = torch.tensor(actions_b, dtype=torch.long)
            rewards_b = torch.tensor(rewards_b, dtype=torch.float32)
            dones_b = torch.tensor(dones_b, dtype=torch.bool)

            # current Q-values for taken actions
            q_values_b = model(states_b)
            q_sa = q_values_b.gather(1, actions_b.view(-1, 1)).squeeze(1)

            # target Q-values using next states (from target network)
            with torch.no_grad():
                next_q_values_b = target_model(next_states_b)
                max_next_q = next_q_values_b.max(dim=1).values
                target_q = rewards_b + gamma * max_next_q * (~dones_b)

            loss = torch.mean((q_sa - target_q) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.detach()

            # periodic hard update of target network
            if global_step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            f"Step {step:03d} | ep={episode:02d} | eps={epsilon:.3f} | "
            f"action={action} | reward={reward:.4f} | loss={float(last_loss):.6f}"
        )

        global_step += 1

        if done:
            episode += 1
            state = preprocess(env.reset())

if __name__ == "__main__":
    run()