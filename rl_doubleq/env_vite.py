import random
from typing import Dict, Tuple

import numpy as np
from playwright.sync_api import sync_playwright


class BigWatermelonEnv:
    """
    Environment wrapper that consumes structured JSON state from the Vite game.
    """

    def __init__(
        self,
        url: str = "http://localhost:5173",
        n_actions: int = 30,
        step_delay: float = 0.3,
        headless: bool = True,
    ) -> None:
        self.url = url
        self.n_actions = n_actions
        self.step_delay = step_delay

        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(
            headless=headless,
            args=["--window-size=360,600"],
        )
        self.page = self.browser.new_page(viewport={"width": 360, "height": 600})
        self.page.goto(self.url)

        self.page.wait_for_selector("canvas")
        self.page.wait_for_function("() => window.__BIGWATERMELON__ !== undefined")

        self.canvas_width: int = int(
            self.page.evaluate("() => window.__BIGWATERMELON__.getCanvasWidth()")
        )
        self.num_fruit_types: int = int(
            self.page.evaluate("() => window.__BIGWATERMELON__.getFruitTypeCount()")
        )
        self.max_fruits: int = 32
        # one-hot current + one-hot next + [score, danger, height]
        # + per-fruit [x, y, radius, type_norm, is_active]
        self.state_dim: int = self.num_fruit_types * 2 + 3 + self.max_fruits * 5

        self.last_score: float = 0.0
        self.last_height_ratio: float = 0.0
        self.last_cluster_metric: float = 0.0

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception:
            pass
        try:
            self._playwright.stop()
        except Exception:
            pass

    def _fetch_raw_state(self) -> Dict:
        return self.page.evaluate("() => window.__BIGWATERMELON__.getState()")

    def _build_state_vector(self, info: Dict) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)

        current_idx = int(info.get("currentFruitIndex", 0))
        upcoming_idx = int(info.get("upcomingFruitIndex", 0))
        score = float(info.get("score", 0.0))
        danger_indicator = float(info.get("dangerIndicator", 0.0))
        height_ratio = float(info.get("heightRatio", 0.0))

        offset = 0
        for i in range(self.num_fruit_types):
            state[offset + i] = 1.0 if i == current_idx else 0.0
        offset += self.num_fruit_types

        for i in range(self.num_fruit_types):
            state[offset + i] = 1.0 if i == upcoming_idx else 0.0
        offset += self.num_fruit_types

        score_norm = min(score / 1000.0, 1.0)
        state[offset] = score_norm
        offset += 1

        state[offset] = danger_indicator
        offset += 1

        state[offset] = height_ratio
        offset += 1

        fruits = info.get("fruits", []) or []
        # Canonicalize ordering: from bottom to top (larger y first in normalized coords)
        fruits = sorted(fruits, key=lambda f: float(f.get("y", 0.0)), reverse=True)
        max_slots = self.max_fruits
        for idx, fruit in enumerate(fruits[:max_slots]):
            base = offset + idx * 5
            state[base] = float(fruit.get("x", 0.0))
            state[base + 1] = float(fruit.get("y", 0.0))
            state[base + 2] = float(fruit.get("radius", 0.0))
            type_index = int(fruit.get("typeIndex", 0))
            if self.num_fruit_types > 1:
                state[base + 3] = type_index / float(self.num_fruit_types - 1)
            else:
                state[base + 3] = 0.0
            # Explicit mask so the network can distinguish empty slots
            state[base + 4] = 1.0

        return state

    @staticmethod
    def _compute_cluster_metric(info: Dict) -> float:
        fruits = info.get("fruits", []) or []
        n = len(fruits)
        if n < 2:
            return 0.0

        metric = 0.0
        for i in range(n):
            fi = fruits[i]
            ti = int(fi.get("typeIndex", 0))
            xi = float(fi.get("x", 0.0))
            yi = float(fi.get("y", 0.0))
            for j in range(i + 1, n):
                fj = fruits[j]
                tj = int(fj.get("typeIndex", 0))
                if tj != ti:
                    continue
                xj = float(fj.get("x", 0.0))
                yj = float(fj.get("y", 0.0))
                dx = xi - xj
                dy = yi - yj
                dist = (dx * dx + dy * dy) ** 0.5
                bonus = max(0.0, 0.3 - dist)
                metric += bonus
        return float(metric)

    def get_state(self) -> np.ndarray:
        info = self._fetch_raw_state()
        return self._build_state_vector(info)

    def reset(self) -> np.ndarray:
        self.page.evaluate("() => window.__BIGWATERMELON__.reset()")
        self.page.wait_for_timeout(300)

        info = self._fetch_raw_state()
        self.last_score = float(info.get("score", 0.0))
        self.last_height_ratio = float(info.get("heightRatio", 0.0))
        self.last_cluster_metric = self._compute_cluster_metric(info)
        return self._build_state_vector(info)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        action_index = int(action_index)
        action_index = max(0, min(self.n_actions - 1, action_index))

        x_ratio = (action_index + 0.5) / self.n_actions
        click_x = x_ratio * float(self.canvas_width)

        self.page.evaluate(
            "(x) => window.__BIGWATERMELON__.act(x)",
            click_x,
        )
        self.page.wait_for_timeout(int(self.step_delay * 1000.0))

        info = self._fetch_raw_state()
        score = float(info.get("score", 0.0))
        height_ratio = float(info.get("heightRatio", 0.0))
        done = bool(info.get("gameOver", False))

        score_diff = score - self.last_score
        height_penalty = 0.1 * height_ratio
        survival_bonus = 0.05

        cluster_metric = self._compute_cluster_metric(info)
        cluster_bonus = cluster_metric - self.last_cluster_metric
        cluster_bonus = float(max(-0.5, min(0.5, cluster_bonus)))

        reward = score_diff + survival_bonus - height_penalty + cluster_bonus
        if done:
            reward += -5.0

        self.last_score = score
        self.last_height_ratio = height_ratio
        self.last_cluster_metric = cluster_metric

        next_state = self._build_state_vector(info)
        return next_state, float(reward), bool(done)

    def sample_action(self) -> int:
        return random.randint(0, self.n_actions - 1)
