# Based on the strategy mentioned in the paper Dynamic RIS partitioning in NOMA systems using deep reinforcement learning
import numpy as np

class SimpleRISEnv:

    def __init__(self, N=100, max_steps=25): # N = No of RIS Surfaces
        self.N = N
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # user 2 distance (will also evolve slightly each step) for now range is 5 to 20 km
        self.dis_user2 = np.random.uniform(5, 20)
        self.partition = np.random.randint(0, self.N + 1)  # N1 in [0, N]
        self.t = 0
        state = np.array([self.dis_user2, self.partition])
        return state
    
    def step(self, action):
        self.t += 1

        N1 = int(action)
        if N1 < 0:
            N1 = 0
        elif N1 > self.N:
            N1 = self.N
        N2 = self.N - N1

        # simple mobility: user 2 moves a bit each step
        self.dis_user2 += np.random.uniform(-0.5, 0.5)
        self.dis_user2 = float(np.clip(self.dis_user2, 5.0, 20.0))

        #fake channel model (Will be replaced with SINR )
        #Data rate of users (throughput)
        R1 = np.log2(1+N1/10)
        R2 = np.log2(1+N2/(self.dis_user2))


        # ECE Replacement components needed..
        # channels = generate_rayleigh_channels()
        # sinr1, sinr2 = compute_sinr(channels, N1, N2)
        # R1 = log2(1 + sinr1)
        # R2 = log2(1 + sinr2)


        reward = (R1+R2)**2 / (2*(R1**2 + R2**2)) # Jain’s Fairness Index
        # 1 = Perfect allocation, 0.5 = unfair allocation (this makes sure that both users get allocated ris elements)
        # Inherently we want both users to get good communication quality hence use this reward function

        self.partition = N1
        state_next = np.array([self.dis_user2, self.partition])
        done = self.t >= self.max_steps
        return state_next, float(reward), done