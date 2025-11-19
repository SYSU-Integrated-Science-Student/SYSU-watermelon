import time
import random
from typing import Tuple

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Convert raw canvas screenshot to a 1x80x80 float32 tensor in [0, 1]."""
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame = frame.astype(np.float32) * (1.0 / 255.0)
    return frame[None, :, :]


class BigWatermelonEnv:
    def __init__(
        self,
        url: str = "http://localhost:5173",
        n_actions: int = 30, 
        step_delay: float = 0.05,
        headless: bool = True,
    ) -> None:
        self.url = url
        self.n_actions = n_actions
        self.step_delay = step_delay

        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(
            headless=headless, 
            args=["--window-size=360,600"]
        )
        self.page = self.browser.new_page(viewport={"width": 360, "height": 600})
        self.page.goto(self.url)

        self.page.wait_for_selector("canvas")
        self.canvas = self.page.query_selector("canvas")

        # [CRITICAL] Ensure the game object is ready before injecting utils
        self.page.wait_for_function("() => window.__BIGWATERMELON__ !== undefined")

        # Inject utils that depend on the EXPOSED variables
        self._inject_js_utils()

        self.canvas_width: int = int(
            self.page.evaluate("window.__BIGWATERMELON__.getCanvasWidth()")
        )
        self.last_score: float = 0.0

    def _inject_js_utils(self):
        """
        Injects helper functions. 
        Crucially, it relies on window.__BIGWATERMELON__.engine being exposed.
        """
        js_code = """
        // Helper to set physics speed
        window.setPhysicsSpeed = (rate) => {
            const mw = window.__BIGWATERMELON__;
            if(mw && mw.engine && mw.engine.timing) {
                mw.engine.timing.timeScale = rate;
            }
        };
        
        // Helper to check stability
        window.isStateStable = () => {
            const mw = window.__BIGWATERMELON__;
            if(!mw || !mw.engine) return true; // Default to true if not found to avoid stuck
            
            // Check all bodies in the Matter.js world
            const bodies = mw.engine.world.bodies;
            // Threshold 0.15 is good for 'mostly stopped'
            return bodies.every(b => b.speed < 0.15);
        };
        """
        self.page.add_script_tag(content=js_code)

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception:
            pass
        try:
            self._playwright.stop()
        except Exception:
            pass

    def get_state(self) -> np.ndarray:
        image_bytes = self.canvas.screenshot()
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return process_frame(frame)

    def reset(self) -> np.ndarray:
        self.page.evaluate("window.__BIGWATERMELON__.reset()")
        # Re-inject just in case reset logic clears globals (though unlikely in Vite)
        self._inject_js_utils()
        time.sleep(0.2)
        
        # Accelerate Physics 5x
        self.page.evaluate("window.setPhysicsSpeed(5.0)")
        
        self.last_score = 0.0
        return self.get_state()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        action_index = int(action_index)
        action_index = max(0, min(self.n_actions - 1, action_index))
        
        # Map discrete action to X coordinate
        x_ratio = (action_index + 0.5) / self.n_actions
        click_x = x_ratio * self.canvas_width
        
        # Execute Action via the exposed API (Cleanest way)
        self.page.evaluate(f"window.__BIGWATERMELON__.act({click_x})")

        # Wait for stability
        for _ in range(40): # Max wait 2s
            if self.page.evaluate("window.isStateStable()"):
                break
            time.sleep(self.step_delay)

        # Fetch info
        info = self.page.evaluate("""() => {
            const mw = window.__BIGWATERMELON__;
            return {
                score: mw.getScore(),
                isOver: mw.isOver()
            }
        }""")
        
        score = float(info['score'])
        done = bool(info['isOver'])

        # Reward Engineering
        reward = (score - self.last_score)
        if done:
            reward = -10.0
        else:
            reward += 0.1 # Survival reward
        
        self.last_score = score
        next_state = self.get_state()

        return next_state, float(reward), bool(done)

    def sample_action(self) -> int:
        return random.randint(0, self.n_actions - 1)