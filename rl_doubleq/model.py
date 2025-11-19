import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif "Linear" in classname:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)

class DQN(nn.Module):
    """
    Simple convolutional Q-network:
    input: (B, 1, 80, 80)
    output: (B, n_actions)
    """
    def __init__(self, num_inputs: int, n_actions: int) -> None:
        super().__init__()
        # Input 80x80
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # -> 40x40
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)         # -> 20x20
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)         # -> 10x10
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)         # -> 5x5

        self.fc1 = nn.Linear(32 * 5 * 5, 256) # 800 -> 256
        self.fc2 = nn.Linear(256, n_actions)

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape=(1, 80, 80)) -> None:
        self.capacity = capacity
        self.state_shape = state_shape
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones