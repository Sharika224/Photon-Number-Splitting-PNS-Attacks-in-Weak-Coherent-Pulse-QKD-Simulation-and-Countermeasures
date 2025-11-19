import numpy as np

class ThresholdDetector:
    def __init__(self, efficiency: float, dark_count: float):
        self.eff = efficiency
        self.p_dark = dark_count

    def click_vector(self, n_arr: np.ndarray) -> np.ndarray:
        # P(click | n) = 1 - (1 - eff)^n ; combine with independent dark
        real_click = (np.random.rand(len(n_arr)) < 1 - (1 - self.eff) ** n_arr)
        dark_click = (np.random.rand(len(n_arr)) < self.p_dark)
        clicks = (real_click | dark_click).astype(int)
        return clicks
