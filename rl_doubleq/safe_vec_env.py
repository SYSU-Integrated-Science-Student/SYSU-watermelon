import multiprocessing as mp
import traceback
from typing import Any, Callable, Iterable, List, Tuple

import cloudpickle
import numpy as np


class CloudpickleWrapper:
    """
    Wraps a callable using cloudpickle so it can be sent to spawned processes.
    """

    def __init__(self, x: Callable[[], Any]) -> None:
        self.x = x

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.x)

    def __setstate__(self, state: bytes) -> None:
        self.x = cloudpickle.loads(state)


def _maybe_accelerate_physics(env: Any, applied_flag: dict) -> None:
    """
    Try to accelerate Matter.js physics inside the child process.

    This should be called after the first successful reset.
    It is safe to call multiple times; the underlying JS is idempotent.
    """
    if applied_flag.get("done"):
        return
    try:
        page = getattr(env, "page", None)
        if page is not None:
            page.evaluate(
                "() => { try { if (window.__BIGWATERMELON__ && "
                "window.__BIGWATERMELON__.engine && "
                "window.__BIGWATERMELON__.engine.timing) { "
                "window.__BIGWATERMELON__.engine.timing.timeScale = 3.0; } } catch (e) {} }"
            )
            applied_flag["done"] = True
    except Exception:
        # We never want physics-accel failures to kill the worker.
        traceback.print_exc()


def _worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:
    """
    Child process worker.

    - Instantiates its own environment (and thus its own Playwright+Chromium).
    - Handles commands from the main process.
    - Ensures the browser is shut down cleanly even on error.
    """
    parent_remote.close()
    env = None
    physics_flag = {"done": False}

    try:
        env = env_fn_wrapper.x()
        while True:
            try:
                cmd, data = remote.recv()
            except (EOFError, OSError, BrokenPipeError):
                break

            try:
                if cmd == "step":
                    action = data
                    next_state, reward, done = env.step(action)
                    if done:
                        next_state = env.reset()
                        _maybe_accelerate_physics(env, physics_flag)
                    try:
                        remote.send((next_state, reward, done))
                    except (EOFError, OSError, BrokenPipeError):
                        break

                elif cmd == "reset":
                    obs = env.reset()
                    _maybe_accelerate_physics(env, physics_flag)
                    try:
                        remote.send(obs)
                    except (EOFError, OSError, BrokenPipeError):
                        break

                elif cmd == "get_state_dim":
                    dim = getattr(env, "state_dim", None)
                    if dim is None:
                        # Fallback: infer from a fresh observation.
                        obs = env.reset()
                        _maybe_accelerate_physics(env, physics_flag)
                        dim = int(np.asarray(obs, dtype=np.float32).shape[-1])
                    try:
                        remote.send(int(dim))
                    except (EOFError, OSError, BrokenPipeError):
                        break

                elif cmd == "close":
                    try:
                        env.close()
                    except Exception:
                        traceback.print_exc()
                    try:
                        remote.close()
                    except Exception:
                        pass
                    break

                else:
                    # Unknown command; report and continue.
                    traceback.print_exc()
            except EOFError:
                break
            except Exception:
                # Log the traceback but try to keep the worker alive so the caller
                # can still attempt a graceful shutdown.
                traceback.print_exc()
    except Exception:
        # Catch any error that happens during env construction.
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()


class SafeVecEnv:
    """
    A simple, robust vectorized environment for heavy, non-picklable envs.

    Key design points:
    - Uses 'spawn' start method to ensure Playwright is only created in children.
    - Each worker process owns its own Playwright+Chromium instance.
    - All IPC is protected with try/except to avoid deadlocks on crashes.
    - Auto-resets environments when done=True, returning the post-reset state
      alongside the terminal flag from the previous step.
    - Performs a best-effort cleanup to avoid zombie Chromium processes.
    """

    def __init__(self, env_fns: Iterable[Callable[[], Any]]) -> None:
        env_fns = list(env_fns)
        if not env_fns:
            raise ValueError("SafeVecEnv requires at least one environment function.")
        if len(env_fns) > 24:
            print("Warning: very high parallelism may cause high CPU load.")

        ctx = mp.get_context("spawn")
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.ps: List[mp.Process] = []
        for work_remote, remote, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, CloudpickleWrapper(fn)),
            )
            p.daemon = False  # We want explicit lifetime management.
            p.start()
            self.ps.append(p)
        for wr in self.work_remotes:
            wr.close()

        self._closed = False

    def get_state_dim(self) -> int:
        """
        Query the first environment for its state dimension.
        """
        if self._closed:
            raise RuntimeError("Cannot query state_dim on closed SafeVecEnv.")

        remote = self.remotes[0]
        try:
            remote.send(("get_state_dim", None))
            dim = remote.recv()
        except (EOFError, OSError, BrokenPipeError) as e:
            raise RuntimeError(f"Failed to get state_dim from worker 0: {e}") from e
        return int(dim)

    def reset(self) -> np.ndarray:
        """
        Reset all environments and return a stacked array of observations.
        """
        if self._closed:
            raise RuntimeError("Cannot reset closed SafeVecEnv.")

        for r in self.remotes:
            r.send(("reset", None))

        results: List[np.ndarray] = []
        for idx, r in enumerate(self.remotes):
            try:
                obs = r.recv()
            except (EOFError, OSError, BrokenPipeError) as e:
                raise RuntimeError(f"Worker {idx} died during reset: {e}") from e
            results.append(np.asarray(obs, dtype=np.float32))
        return np.stack(results, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments with the provided actions.

        Returns:
            next_states: (num_envs, state_dim) float32
            rewards:     (num_envs,) float32
            dones:       (num_envs,) bool
        """
        if self._closed:
            raise RuntimeError("Cannot step closed SafeVecEnv.")

        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got shape {actions.shape}."
            )

        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))

        next_states: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []

        for idx, r in enumerate(self.remotes):
            try:
                ns, rew, done = r.recv()
            except (EOFError, OSError, BrokenPipeError) as e:
                raise RuntimeError(f"Worker {idx} died during step: {e}") from e

            next_states.append(np.asarray(ns, dtype=np.float32))
            rewards.append(float(rew))
            dones.append(bool(done))

        return (
            np.stack(next_states, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
        )

    def close(self) -> None:
        """
        Close all environments and terminate the worker processes.
        """
        if self._closed:
            return

        # Ask workers to close their envs/browsers.
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (EOFError, OSError, BrokenPipeError):
                # Worker might already be dead.
                pass

        # Join with a timeout, then forcefully terminate if needed.
        for idx, p in enumerate(self.ps):
            p.join(timeout=5.0)
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    traceback.print_exc()

        self._closed = True

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self) -> "SafeVecEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def make_env(
    url: str,
    n_actions: int,
    step_delay: float = 0.1,
    headless: bool = True,
) -> Callable[[], Any]:
    """
    Environment factory.

    NOTE: This function deliberately imports the environment implementation
    inside the returned thunk so that Playwright is only imported inside
    child processes, never in the main process.
    """

    def _thunk():
        from .env_vite import BigWatermelonEnv

        return BigWatermelonEnv(
            url=url,
            n_actions=n_actions,
            step_delay=step_delay,
            headless=headless,
        )

    return _thunk


def _example_training_loop() -> None:
    """
    Minimal example of using SafeVecEnv with a DQN + ReplayBuffer.
    This is meant as a reference and is not heavily optimized.
    """
    import time

    import torch

    from .model import DQN, ReplayBuffer

    url = "http://localhost:5173"
    n_actions = 30
    num_envs = 4  # 4–6 is a good range for most machines.

    ctx = mp.get_context("spawn")
    try:
        ctx.set_start_method("spawn")
    except RuntimeError:
        # Already set; ignore.
        pass

    env_fns = [make_env(url, n_actions, step_delay=0.1, headless=True) for _ in range(num_envs)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with SafeVecEnv(env_fns) as vec_env:
        state_dim = vec_env.get_state_dim()

        q_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
        target_net = DQN(state_dim=state_dim, n_actions=n_actions).to(device)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
        replay_buffer = ReplayBuffer(capacity=100_000, state_dim=state_dim)
        replay_buffer.device = device

        gamma = 0.99
        batch_size = 64
        total_steps = 10_000
        target_update = 1_000

        obs = vec_env.reset()
        global_step = 0

        start_time = time.time()
        while global_step < total_steps:
            epsilon = max(0.05, 1.0 - global_step / 50_000.0)

            actions: List[int] = []
            for i in range(num_envs):
                if np.random.rand() < epsilon:
                    a = np.random.randint(0, n_actions)
                else:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(obs[i]).unsqueeze(0).to(device)
                        q_vals = q_net(s_tensor)
                        a = int(q_vals.argmax(dim=1).item())
                actions.append(a)

            next_obs, rewards, dones = vec_env.step(np.asarray(actions, dtype=np.int64))

            for i in range(num_envs):
                replay_buffer.add(
                    obs[i],
                    actions[i],
                    float(rewards[i]),
                    next_obs[i],
                    bool(dones[i]),
                )

            obs = next_obs
            global_step += num_envs

            # Learning step
            if replay_buffer.can_sample(batch_size):
                states, actions_b, rewards_b, next_states, dones_b = replay_buffer.sample(batch_size)

                q_values = q_net(states)
                state_action_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_online = q_net(next_states)
                    next_actions_b = next_q_online.argmax(dim=1)
                    next_q_target = target_net(next_states)
                    next_state_values = next_q_target.gather(
                        1, next_actions_b.unsqueeze(1)
                    ).squeeze(1)
                    expected_state_action_values = rewards_b + gamma * (
                        1.0 - dones_b
                    ) * next_state_values

                loss = torch.nn.functional.smooth_l1_loss(
                    state_action_values, expected_state_action_values
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                if global_step % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[train] step={global_step} "
                        f"loss={loss.item():.4f} "
                        f"elapsed={elapsed:.1f}s"
                    )

            if global_step % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())


if __name__ == "__main__":
    _example_training_loop()
