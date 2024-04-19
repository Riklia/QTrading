from dataclasses import dataclass


@dataclass(slots=True)
class EnvParameters:
    data_path: str
    start_time: int
    end_time: int
    # how many previous prices take into consideration
    window: int
    initial_balance: float
    # balance logs from environment - experimental feature
    record_balance: bool = False
    # set step (0, 1) if you want to be able to sell/buy using some percentage of current balance
    action_step_size: float = 1
    # if current balance < initial balance * terminate_threshold => episode is terminated
    terminate_threshold: float = 0.05
    transaction_fee: float = 0.01
