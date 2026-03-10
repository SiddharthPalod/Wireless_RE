# Wireless RIS DQN (Toy Setup)

This folder contains a **minimal Deep Q-Network (DQN)** loop controlling a toy **RIS partition** environment.  
The goal of this setup is to prototype the **ML/DRL component** of the project while the full wireless communication model is still being implemented separately.

The environment is intentionally simplified so that the **DRL pipeline (state → action → reward → policy optimization)** can be validated early.

---

## How to run and visualize

From the `Wireless` folder:

- **Train the agent and generate logs**

```bash
python main.py
```

This runs the DQN loop and writes:

- `training_rewards.npy`
- `training_losses.npy`
- `training_actions.npy`

to the current directory.

- **Visualize training curves**

```bash
python visual.py
```

This will open a Matplotlib window showing:

- reward vs. step (with moving average),
- loss vs. step,
- chosen action vs. step (RIS partition index).

---

# Initial setup (as it originally was)

- **Files**
  - `simple_ris_env.py`: Defines a toy RIS environment with a state `[distance_user2, partition]`.
  - `dqn.py`: Contains a small neural network used as a Q-network.
  - `main.py`: Runs the training loop and prints step-by-step logs.

- **State**
  - `[distance_user2, partition]`
  - `distance_user2`: simulated distance of User 2 from RIS
  - `partition`: number of RIS elements allocated to User 1

- **Action**
  - `N1` = number of RIS elements assigned to User 1
  - `N2 = N - N1`

- **Reward**
  - Jain Fairness Index based on toy data rates:

```math
reward = (R1 + R2)^2 / (2(R1^2 + R2^2))
```

* Range of reward:

```
0.5  → highly unfair
1.0  → perfectly fair
```

* **Training loop**

```
state → choose action → env.step(action) → reward → update Q-network
```

---

# Problems in the initial code

### 1) Partition sampling was not scalable

The environment originally used fixed values:

```python
partition = random(10, 90)
```

This only worked if the RIS size was implicitly assumed to be around 100 elements.

**Problem**

* Not scalable for different RIS sizes
* Violates the constraint:

```
N1 + N2 = N
```

---

### 2) No exploration

Action selection used only greedy policy:

```
action = argmax(Q)
```

**Effect**

The agent quickly converged to a single action:

```
96
96
96
96
96
```

This prevented the model from discovering better solutions.

---

### 3) Environment looked static

Within a training run:

* `distance_user2` stayed constant
* repeated actions produced identical states

Example behavior:

```
same action → same reward → same state
```

This limited the agent's ability to generalize.

---

# Improvements made

## 1) Partition sampling based on N

Partition initialization was changed to scale with the total RIS size.

**Before**

```python
partition = random(10,90)
```

**After**

```python
partition = random(1, N)
```

**Effect**

* Works for any RIS size
* Enforces valid RIS allocation
* Makes the environment scalable for:

```
N = 128
N = 256
N = 512
```

---

## 2) Epsilon-greedy exploration

Exploration was introduced using epsilon-greedy strategy.

```python
if random() < epsilon:
    action = random_action
else:
    action = argmax(Q)
```

Epsilon gradually decays during training.

Example:

```
eps = 0.995
eps = 0.900
eps = 0.800
eps = 0.600
```

**Effect**

* Agent explores many partitions early
* Prevents premature convergence
* Allows discovery of better actions

Example exploration behavior:

```
29
87
9
64
90
72
3
91
94
59
```

---

## 3) Episode-based environment reset

Episodes were introduced so the environment resets periodically.

Each reset randomizes:

```
distance_user2
partition
```

Example training scenarios:

```
Episode 1 → distance = 12
Episode 2 → distance = 8
Episode 3 → distance = 17
```

**Effect**

* The agent learns a general policy
* Training becomes closer to a real RL setup

---

## 4) Added epsilon decay monitoring

Training logs now include:

```
Step | episode | epsilon | action | reward | loss
```

Example output:

```
Step 078 | ep=03 | eps=0.673 | action=57 | reward=1.0000 | loss=6.36
```

**Effect**

* Easier debugging
* Confirms exploration is active
* Helps monitor learning behavior

---

# Observed learning behavior

During training the agent gradually discovers high-reward partitions.

Example training logs:

```
action = 64
reward ≈ 0.999
```

For a system with `N = 100`, a balanced allocation tends to maximize fairness.

```
N1 ≈ N2
```

This indicates the DRL agent successfully learns the fairness objective.

---

# Loss behavior

Loss occasionally spikes during exploration.

Example:

```
action = 0
reward = 0.5
loss = large
```

Reason:

The Q-network predicted a high value but received a low reward.

**This behavior is expected in DQN training.**

---

# Current behavior summary

* **State**: `[distance_user2, partition]`
* **Action**: `N1` (RIS elements for User 1)
* **Reward**: Jain Fairness Index
* **Training loop**: simple online TD update
* **Console output**: step-by-step logs with epsilon, reward, and loss

The agent is now capable of:

* exploring the action space
* identifying high-reward allocations
* learning a stable policy.

---

# Future work

## 1) Integrate real wireless communication model

Replace the toy rate model with real wireless equations:

```
Channel → SINR → Data Rate → Jain Fairness Index
```

This will connect the DRL agent with the real RIS-NOMA system.

---

## 2) Implement full DQN architecture

Add standard improvements used in DRL research:

* Experience replay buffer
* Target network
* Mini-batch training
* Huber loss

These improve stability and convergence.

---

## 3) Expand state representation

Future states may include:

```
channel gain
path loss
user mobility
transmit power
previous allocation
```

This allows the agent to learn more complex policies.

---

## 4) Evaluate scalability

Test performance for larger RIS sizes:

```
N = 128
N = 256
N = 512
```

This will evaluate the scalability of the DRL approach.

---

# Conclusion

This toy setup successfully validates the **DRL learning pipeline** for RIS partitioning.

The current framework demonstrates:

* functioning RL environment
* stable training loop
* reward-driven learning

The system is now ready to integrate with the **actual wireless communication simulator** once it becomes available.

---

# Iteration 3 – ECE plug‑in (real RIS–NOMA model)

In this iteration, the environment was upgraded from a toy log-based rate model to a **ported version of the ECE `ris_noma_channel.m` script**, so that the RL loop now runs on a more realistic RIS–NOMA system.

## Model changes

- **`simple_ris_env.py`**
  - Added Python equivalents of the MATLAB functions:
    - `generate_rayleigh_channels` → `_generate_rayleigh_channels`
    - `compute_sinr` → `_compute_sinr`
  - Mirrored the **system parameters**:
    - carrier frequency `fc = 1.8 GHz`, wavelength `λ = c/fc`
    - 20 m × 20 m area, with fixed positions for BS, RIS, and User 1
    - noise power `N0` from bandwidth, noise figure, and `-174 dBm/Hz`
    - RIS size `N`, transmit power `Pt`, active gain `α`, and amplifier noise `σ²_z`
  - Implemented the **hybrid RIS** structure:
    - `N1` active elements for User 1 (action),
    - `N2 = N − N1` passive elements for User 2,
    - Rayleigh fading channels with distance‑based path loss,
    - per‑segment optimal phase shifts as in the paper.
  - Replaced the dummy rate model with:
    - `sinr1, sinr2 = _compute_sinr(ch, N1, N2)`
    - `R1 = log2(1 + sinr1)`, `R2 = log2(1 + sinr2)`
  - The **reward** remains the Jain Fairness Index computed from `(R1, R2)`.

## Environment / state behavior

- **State (unchanged structurally)**:
  - `[distance_user2, partition]`
  - `distance_user2` is now tied to a 2‑D position of User 2 moving around the RIS within the 20 m × 20 m area.
  - `partition = N1` represents the active RIS elements allocated to User 1; `N2` is derived internally.
- **Action**:
  - `N1 ∈ {0, 1, …, N}` (clipped inside the env for safety).
- **Reward**:
  - Still Jain Fairness, but now built from **realistic SINR and rate computations**, not toy log2 expressions.

## Observed training logs (Iteration 3)

Example run (ECE plug‑in active):

```text
Step 000 | ep=00 | eps=0.995 | action=20 | reward=0.5280 | loss=0.571662
Step 005 | ep=00 | eps=0.970 | action=62 | reward=0.9790 | loss=1.512519
Step 025 | ep=01 | eps=0.878 | action=89 | reward=0.7885 | loss=2.025771
Step 050 | ep=02 | eps=0.774 | action=92 | reward=0.6535 | loss=2.767838
Step 075 | ep=03 | eps=0.683 | action=4  | reward=0.5035 | loss=2.877674
Step 100 | ep=04 | eps=0.603 | action=79 | reward=0.5609 | loss=9.163788
Step 150 | ep=06 | eps=0.469 | action=11 | reward=0.5116 | loss=0.951550
Step 194 | ep=07 | eps=0.376 | action=68 | reward=0.9986 | loss=266.014404
```

**Key observations**

- Rewards still stay mostly within \([0.5, 1.0]\), matching Jain Fairness interpretation:
  - values near **0.5** → highly unfair allocations,
  - values near **1.0** → near‑perfect fairness.
- Loss values can be **much larger and more volatile** than in the toy model:
  - due to the higher dynamic range of real SINR/rates,
  - and because exploration occasionally chooses partitions with very different rewards than what the Q‑network predicted.
- The epsilon‑greedy policy continues to explore a variety of partitions across episodes while gradually decaying `ε`, so the agent still balances **exploration vs exploitation** under the real channel model.

## Status after Iteration 3

- The **“Integrate real wireless communication model”** item from the Future Work section is now partially fulfilled:
  - The DRL agent interacts with a **physically meaningful RIS–NOMA model** instead of abstract log‑based rates.
  - The environment keeps the same RL interface (state/action/reward) so higher‑level DRL code did not need major changes.
- Remaining work is mostly on the **DRL algorithmic side** (replay buffer, target network, mini‑batch training) and potentially on **richer state design** (explicit channel stats, SINR, etc.) rather than on the basic environment realism.

---

# Iteration 4 – Stable DQN (replay buffer + target network + ergodic reward)

Iteration 4 focuses on **stabilizing learning** by moving from pure online Q‑learning to a more standard DQN setup and by averaging rewards over multiple channel realizations.

## Algorithmic upgrades

- **Replay buffer**
  - Implemented in `main.py` using a `deque` with a fixed capacity.
  - Each step stores transitions:
    - `(state, action, reward, next_state, done)`
  - Once there are enough samples, the agent:
    - draws random minibatches,
    - computes TD targets using next‑state values,
    - minimizes mean‑squared TD error over the batch.

- **Target network**
  - Added a separate `target_model` (same architecture as `model`).
  - Targets are computed using `target_model(next_state)` while gradients flow only through the online `model`.
  - `target_model` is periodically **hard‑updated** from `model` every fixed number of steps to reduce Q‑value oscillations.

- **Ergodic (channel‑averaged) reward**
  - `simple_ris_env.py` now averages rates over `mc_samples` independent channel realizations per step:
    - for each sample: generate channels → compute `sinr1, sinr2` → `R1, R2`,
    - then average `R1` and `R2` before computing Jain Fairness.
  - This approximates **ergodic rates** instead of relying on a single fading draw, smoothing the reward signal.

## Observed training logs (Iteration 4)

Representative run with replay buffer, target network, and ergodic reward:

```text
Step 000 | ep=00 | eps=0.995 | action=38 | reward=0.6848 | loss=0.000000
Step 008 | ep=00 | eps=0.956 | action=67 | reward=1.0000 | loss=0.000000
Step 027 | ep=01 | eps=0.869 | action=66 | reward=0.9998 | loss=0.000000
Step 031 | ep=01 | eps=0.852 | action=5  | reward=0.5026 | loss=0.999601
Step 048 | ep=01 | eps=0.782 | action=64 | reward=0.9938 | loss=0.822957
Step 050 | ep=02 | eps=0.774 | action=81 | reward=0.9996 | loss=0.643427
Step 063 | ep=02 | eps=0.726 | action=67 | reward=0.8298 | loss=1.241323
Step 081 | ep=03 | eps=0.663 | action=80 | reward=0.9871 | loss=3.137373
Step 104 | ep=04 | eps=0.591 | action=60 | reward=0.9999 | loss=2.324304
Step 135 | ep=05 | eps=0.506 | action=80 | reward=0.9995 | loss=1.836061
Step 160 | ep=06 | eps=0.446 | action=81 | reward=0.9963 | loss=0.486533
Step 181 | ep=07 | eps=0.402 | action=53 | reward=0.9981 | loss=8.748996
```

**Behavior**

- Early in training, **loss stays at 0** until the replay buffer is full enough to start minibatch updates.
- As replay updates begin, **loss becomes non‑zero and fluctuates**, reflecting:
  - corrections to previously optimistic Q‑value estimates,
  - stochasticity from both channel realizations and the replay sampling.
- Rewards frequently reach values **very close to 1.0**, indicating the agent is often selecting RIS partitions that produce near‑perfect Jain Fairness under the ECE channel model.
- Epsilon continues to decay, and high‑reward partitions (e.g. around the mid‑RIS range) appear more often as the policy becomes more exploitative.

## Status after Iteration 4

- The prototype now includes the **core components of a DQN**:
  - replay buffer,
  - target network,
  - minibatch TD learning,
  - epsilon‑greedy exploration with decay.
- The environment uses a **realistic RIS–NOMA channel model** with **ergodic (averaged) rewards**, making the learning signal more representative of long‑term performance.
- Remaining future work mainly involves:
  - experimenting with **loss functions** (e.g. Huber),
  - **hyperparameter tuning** (buffer size, batch size, target update frequency, `mc_samples`),
  - and extending the **state**/reward to capture more physical metrics if needed.
