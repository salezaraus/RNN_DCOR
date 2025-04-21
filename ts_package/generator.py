"""
Time series data generation wrapper.
"""
from typing import List, Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import data_generator as dg


class TSGenerator:
    """
    Wrapper for various simulated time series (AR, MA, GARCH, etc.).
    """
    def __init__(self,
                 dataset: int,
                 length: int,
                 num_ts: int,
                 mu: float,
                 sigma: float,
                 coef_lag_list: List[Tuple[float,int]],
                 beta_list: List[Tuple[float,int]],
                 omega: float,
                 noise_frequency: float,
                 drift: float):
        self.dataset = dataset
        self.length = length
        self.num_ts = num_ts
        self.mu = mu
        self.sigma = sigma
        self.coef_lag_list = coef_lag_list
        self.beta_list = beta_list
        self.omega = omega
        self.noise_frequency = noise_frequency
        self.drift = drift
        # generate series
        self.ts = self._generate()

    def _generate(self) -> torch.Tensor:
        """Call underlying data_generator functions based on `dataset`."""
        if self.dataset == 1:
            return dg.dataset1(self.length, self.num_ts,
                               self.noise_frequency,
                               self.mu, self.sigma)
        # ... handle other cases similarly ...
        elif self.dataset == 12:
            return dg.dataset_GARCH(self.length, self.coef_lag_list,
                                    self.beta_list, self.omega,
                                    self.mu, self.sigma)
        else:
            raise ValueError(f"Unknown dataset code: {self.dataset}")

    def create_windows(self,
                       window: int,
                       horizon: int,
                       overlap: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unfold time series into sliding windows and targets.
        Returns: (windows, targets)
        """
        # similar to original generate_data + sample_data
        # ... implementation here ...
        pass