from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from src.environment.env_parameters import EnvParameters


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
    save_plot_every: int = 0


@dataclass(slots=True)
class TrainConfig:
    model_dir: str
    model_name: str
    # after post initialization, this will be ModelParameters
    learning_parameters: LearningParameters
    # after post initialization, this will be EnvParameters
    env_parameters: EnvParameters
    overwrite: bool = False

    def __post_init__(self):
        self.learning_parameters = LearningParameters(**self.learning_parameters)
        self.env_parameters = EnvParameters(**self.env_parameters)
        model_dir = Path(self.model_dir)
        if model_dir.is_dir() and not self.overwrite:
            timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.model_dir += f"_{timestamp_string}"
            model_dir = Path(self.model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir/"run_parameters.log", "w") as parameters_logfile:
            parameters_logfile.write(str(self))
