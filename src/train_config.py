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
    overwrite: bool = False
    continue_learning: bool = False
    load_model_directory: Path | None = None


@dataclass(slots=True)
class QTradingConfigurations:
    model_dir: Path
    model_name: str
    # mode: one of "offline_training", "use"
    mode: str
    # after post initialization, this will be EnvParameters
    env_parameters: EnvParameters
    # after post initialization, this will be ModelParameters or None if in use mode
    learning_parameters: LearningParameters | None = None

    def _add_timestamp_to_dirname(self):
        timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_dir = self.model_dir.parent / f"{self.model_dir.name}_{timestamp_string}"

    def _process_use_mode(self):
        self.learning_parameters = None
        if not self.model_dir.is_dir():
            raise NotADirectoryError(f"{self.model_dir.resolve()} is not a directory")

    def _process_learning_mode(self):
        self.learning_parameters = LearningParameters(**self.learning_parameters)
        if self.learning_parameters.continue_learning:
            if not self.model_dir.is_dir():
                raise NotADirectoryError(f"{self.model_dir.resolve()} is not a directory")

            self.learning_parameters.load_model_directory = self.model_dir
            if not self.learning_parameters.overwrite:
                self._add_timestamp_to_dirname()
        else:
            # new learning
            self.learning_parameters.load_model_directory = None
            if self.model_dir.is_dir() and not self.learning_parameters.overwrite:
                self._add_timestamp_to_dirname()

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        allowed_modes = ["offline_training", "use"]
        if self.mode not in allowed_modes:
            raise NotImplementedError(f"Mode: expected one of {allowed_modes}")

        self.env_parameters = EnvParameters(**self.env_parameters)
        self.model_dir = Path(self.model_dir)
        if self.mode == "use":
            self._process_use_mode()
            return
        if self.learning_parameters is None:
            raise ValueError("Expected learning parameters, got None")

        self._process_learning_mode()
        with open(self.model_dir / "run_parameters.txt", "w") as parameters_file:
            parameters_file.write(str(self))

