import gymnasium as gym
from enum import Enum


class MainActionTypes(Enum):
    HOLD = 1
    LONG = 2
    SHORT = 3

    @staticmethod
    def percentage_to_type(val: float):
        if val > 0:
            return MainActionTypes.LONG
        elif val < 0:
            return MainActionTypes.SHORT
        else:
            return MainActionTypes.HOLD

    def get_action_value(self) -> int:
        if self.value == MainActionTypes.HOLD:
            return 0
        elif self.value == MainActionTypes.LONG:
            return 1
        else:
            return -1


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
