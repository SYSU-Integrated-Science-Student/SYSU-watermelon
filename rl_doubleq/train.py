import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .env_vite import BigWatermelonEnv
from .model import DQN, ReplayBuffer


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a (Double) DQN agent for bigwatermelon")
    parser.add_argument("--url", type=str, default="http://localhost:5173")
    parser.add_argument("--n_actions", type=int, default=30)
    parser.add_argument("--step_delay", type=float, default=0.05)
    parser.add_argument("--headless", action="store_true", help="Run browser without UI")

    parser.add_argument("--buffer_capacity", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_final", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=int, default=100000)

    parser.add_argument("--target_update", type=int, default=1000, help="Target network update frequency (steps)")
    parser.add_argument("--max_episodes", type=int, default=500)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)

    parser.add_argument("--save_dir", type=str, default="models_doubleq")
    parser.add_argument("--save_interval", type=int, default=50, help="Save every N episodes")

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
        # Double DQN: action from online net, value from target net
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = BigWatermelonEnv(
        url=args.url,
        n_actions=args.n_actions,
        step_delay=args.step_delay,
        headless=args.headless,
    )

    q_net = DQN(num_inputs=1, n_actions=args.n_actions).to(device)
    target_net = DQN(num_inputs=1, n_actions=args.n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(args.buffer_capacity, state_shape=(1, 80, 80))
    replay_buffer.device = device

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0

    try:
        for ep in range(1, args.max_episodes + 1):
            state = env.reset()
            episode_reward = 0.0

            for t in range(1, args.max_steps_per_episode + 1):
                epsilon = linear_epsilon(
                    global_step, args.epsilon_start, args.epsilon_final, args.epsilon_decay
                )

                if random.random() < epsilon:
                    action = random.randrange(args.n_actions)
                else:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                        q_vals = q_net(s_tensor)
                        action = int(q_vals.argmax(dim=1).item())

                next_state, reward, done = env.step(action)

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                global_step += 1

                loss = train_step(
                    q_net=q_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                )

                if global_step % args.target_update == 0 and global_step > 0:
                    target_net.load_state_dict(q_net.state_dict())

                if global_step % 100 == 0:
                    print(
                        f"[train] step={global_step} ep={ep} t={t} "
                        f"epsilon={epsilon:.3f} reward={reward:.3f} loss={loss:.4f}"
                    )

                if done:
                    break

            print(f"[train] Episode {ep} finished: total_reward={episode_reward:.2f}")

            if ep % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f"doubleq_ep{ep}.pt")
                torch.save(q_net.state_dict(), save_path)
                print(f"[train] Saved checkpoint to {save_path}")

        final_path = os.path.join(args.save_dir, "doubleq_final.pt")
        torch.save(q_net.state_dict(), final_path)
        print(f"[train] Training complete, saved final model to {final_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
