import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .model import DQN, ReplayBuffer
from .safe_vec_env import SafeVecEnv, make_env


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel Double DQN training for the BigWatermelon game using SafeVecEnv."
    )

    # Environment configuration
    parser.add_argument("--url", type=str, default="http://localhost:5173")
    parser.add_argument("--n_actions", type=int, default=30)
    parser.add_argument("--step_delay", type=float, default=0.1)
    parser.add_argument("--headless", action="store_true", help="Run browser without UI")
    parser.add_argument("--num_envs", type=int, default=16)

    # Replay buffer / optimization
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--updates_per_step", type=int, default=2)

    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_final", type=float, default=0.05)
    parser.add_argument(
        "--epsilon_decay",
        type=int,
        default=100_000,
        help="Number of environment steps over which epsilon is annealed.",
    )

    # Training schedule
    parser.add_argument(
        "--total_steps",
        type=int,
        default=300_000,
        help="Total number of environment steps across all envs.",
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=10_000,
        help="How many environment steps between target network updates.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="Log training stats every this many environment steps.",
    )

    # Saving
    parser.add_argument("--save_dir", type=str, default="models_doubleq")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50_000,
        help="Save a checkpoint every this many environment steps.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=1)

    return parser


def linear_epsilon(step: int, start: float, final: float, decay: int) -> float:
    if decay <= 0:
        return final
    epsilon = start - (start - final) * (step / float(decay))
    return float(max(final, epsilon))


def train_step(
    q_net: DQN,
    target_net: DQN,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
) -> float:
    if not replay_buffer.can_sample(batch_size):
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    q_values = q_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: action from online net, value from target net.
        next_q_online = q_net(next_states)
        next_actions = next_q_online.argmax(dim=1)

        next_q_target = target_net(next_states)
        next_state_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        expected_state_action_values = rewards + gamma * (1.0 - dones) * next_state_values

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Force CPU usage with limited intra-op threads
    torch.set_num_threads(4)
    device = torch.device("cpu")

    # Build vectorized environment
    env_fns = [
        make_env(args.url, args.n_actions, args.step_delay, args.headless)
        for _ in range(args.num_envs)
    ]

    os.makedirs(args.save_dir, exist_ok=True)

    with SafeVecEnv(env_fns) as vec_env:
        state_dim = vec_env.get_state_dim()

        q_net = DQN(state_dim=state_dim, n_actions=args.n_actions).to(device)
        target_net = DQN(state_dim=state_dim, n_actions=args.n_actions).to(device)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

        replay_buffer = ReplayBuffer(args.buffer_capacity, state_dim=state_dim)
        replay_buffer.device = device

        states = vec_env.reset()
        episode_returns = np.zeros(args.num_envs, dtype=np.float32)
        recent_returns: List[float] = []

        global_steps = 0  # Count of environment steps across all envs
        next_log_step = args.log_interval
        next_save_step = args.save_interval
        last_loss: float = 0.0

        start_time = time.time()

        while global_steps < args.total_steps:
            epsilon = linear_epsilon(
                step=global_steps,
                start=args.epsilon_start,
                final=args.epsilon_final,
                decay=args.epsilon_decay,
            )

            # Select actions for each environment
            actions = np.empty(args.num_envs, dtype=np.int64)
            for i in range(args.num_envs):
                if np.random.rand() < epsilon:
                    actions[i] = np.random.randint(args.n_actions)
                else:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(states[i]).unsqueeze(0).to(device)
                        q_vals = q_net(s_tensor)
                        actions[i] = int(q_vals.argmax(dim=1).item())

            next_states, rewards, dones = vec_env.step(actions)

            # Store transitions and track episode returns
            for i in range(args.num_envs):
                replay_buffer.add(
                    states[i],
                    int(actions[i]),
                    float(rewards[i]),
                    next_states[i],
                    bool(dones[i]),
                )
                episode_returns[i] += float(rewards[i])
                if dones[i]:
                    recent_returns.append(float(episode_returns[i]))
                    if len(recent_returns) > 100:
                        recent_returns = recent_returns[-100:]
                    episode_returns[i] = 0.0

            states = next_states
            global_steps += args.num_envs

            # Gradient updates
            for _ in range(args.updates_per_step):
                last_loss = train_step(
                    q_net=q_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                )

            # Target network sync
            if global_steps >= args.target_update and (global_steps % args.target_update) < args.num_envs:
                target_net.load_state_dict(q_net.state_dict())

            # Logging
            if global_steps >= next_log_step:
                elapsed = time.time() - start_time
                steps_per_sec = global_steps / max(elapsed, 1e-6)
                if recent_returns:
                    avg_return_20 = float(np.mean(recent_returns[-20:]))
                else:
                    avg_return_20 = 0.0

                print(
                    f"[train] steps={global_steps} "
                    f"epsilon={epsilon:.3f} "
                    f"loss={last_loss:.4f} "
                    f"avg_return_20={avg_return_20:.2f} "
                    f"steps/s={steps_per_sec:.1f}"
                )
                next_log_step += args.log_interval

            # Checkpoint saving
            if global_steps >= next_save_step:
                ckpt_path = os.path.join(args.save_dir, f"doubleq_steps{global_steps}.pt")
                torch.save(q_net.state_dict(), ckpt_path)
                print(f"[train] Saved checkpoint to {ckpt_path}")
                next_save_step += args.save_interval

        final_path = os.path.join(args.save_dir, "doubleq_final.pt")
        torch.save(q_net.state_dict(), final_path)
        print(f"[train] Training complete, saved final model to {final_path}")


if __name__ == "__main__":
    main()
