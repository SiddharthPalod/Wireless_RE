import torch
from dqn import DQN
from simple_ris_env import SimpleRISEnv


def run():
    env = SimpleRISEnv()

    state_dim = 2  # [dis_user2, partition]
    n_actions = env.N + 1  # N1 in [0, N]

    model = DQN(state_dim=state_dim, n_actions=n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    def preprocess(state):
        # normalize to roughly [0,1] ranges for stability
        dis_user2 = float(state[0]) / 20.0
        partition = float(state[1]) / float(env.N)
        return torch.FloatTensor([dis_user2, partition])

    state = env.reset()
    state = preprocess(state)
    episode = 0

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

        # compute target using one-step TD
        with torch.no_grad():
            next_q_values = model(state_next_tensor)
            target = reward + gamma * torch.max(next_q_values)

        loss = (q_values[action] - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = state_next_tensor

        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Step {step:03d} | ep={episode:02d} | eps={epsilon:.3f} | action={action} | reward={reward:.4f} | loss={loss.item():.6f}")

        if done:
            episode += 1
            state = preprocess(env.reset())

if __name__ == "__main__":
    run()