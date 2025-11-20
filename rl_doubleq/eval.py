import argparse
import os

import numpy as np
import torch

from .env_vite import BigWatermelonEnv
from .model import DuelingDQN


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained Double DQN agent")
    parser.add_argument("--url", type=str, default="http://localhost:5173")
    parser.add_argument("--n_actions", type=int, default=30)
    parser.add_argument("--step_delay", type=float, default=0.3)
    parser.add_argument("--headless", action="store_true", help="Run without UI")

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("models_doubleq", "doubleq_final.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument("--episodes", type=int, default=5)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    device = torch.device("cpu")

    env = BigWatermelonEnv(
        url=args.url,
        n_actions=args.n_actions,
        step_delay=args.step_delay,
        headless=args.headless,
    )

    q_net = DuelingDQN(state_dim=env.state_dim, n_actions=args.n_actions).to(device)
    q_net.load_state_dict(torch.load(args.model_path, map_location=device))
    q_net.eval()

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            env.page.evaluate(
                "() => { if(window.__BIGWATERMELON__?.engine?.timing) "
                "window.__BIGWATERMELON__.engine.timing.timeScale = 3.0; }"
            )
            episode_reward = 0.0
            steps = 0

            while True:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = q_net(s_tensor)
                    action = int(q_values.argmax(dim=1).item())

                next_state, reward, done = env.step(action)
                episode_reward += reward
                steps += 1
                print(f"[Eval] step={steps} action_index={action}")
                state = next_state

                if done:
                    print(
                        f"[Eval] Episode {ep} finished: "
                        f"steps={steps}, total_reward={episode_reward:.2f}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
