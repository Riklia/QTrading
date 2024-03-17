from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LearningParameters:
    gamma: float = 0.97
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 64
    num_episodes: int = 10000
    print_frequency: int = 10
    render: bool = False,


@dataclass(slots=True)
class EnvParameters:
    data_path: str
    max_time_point: int
    # how many previous prices take into consideration
    window: int
    initial_balance: float
    record_balance: bool = False
    # set step (0, 1) if you want to be able to sell/buy using some percentage of current balance
    action_step_size: float = 1


@dataclass(slots=True)
class TrainConfig:
    model_dir: str
    model_name: str
    # after post initialization, this will be ModelParameters
    learning_parameters: LearningParameters
    # after post initialization, this will be EnvParameters
    env_parameters: EnvParameters

    def __post_init__(self):
        self.learning_parameters = LearningParameters(**self.learning_parameters)
        self.env_parameters = EnvParameters(**self.env_parameters)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


