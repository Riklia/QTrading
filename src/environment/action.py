import gymnasium as gym
from enum import Enum


class MainActionTypes(Enum):
    HOLD = 1
    SELL = 2
    BUY = 3

    @staticmethod
    def percentage_to_type(val: float):
        if val > 0:
            return MainActionTypes.SELL
        elif val < 0:
            return MainActionTypes.BUY
        else:
            return MainActionTypes.HOLD


class RangeSpace(gym.spaces.Discrete):
    """
    RangeSpace allows to create a discrete space from a continuous range [low, high]
    """
    def __init__(self, low: float, high: float, step: float):
        num_actions = int((high - low) / step) + 1
        super(RangeSpace, self).__init__(num_actions)
        self.low = low
        self.high = high
        self.step = step

    def range_value(self, discrete_value: int):
        return self.low + self.step * discrete_value

    def __repr__(self):
        return f"RangeSpace(low={self.low}, high={self.high}, step={self.step})"
