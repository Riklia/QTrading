from dataclasses import dataclass


@dataclass(slots=True)
class EnvParameters:
    data_path: str
    start_time: int
    end_time: int
    # how many previous prices take into consideration
    window: int
    # balance logs from environment - experimental feature, because now logs are hard to understand
    record_balance: bool = False
    transaction_fee: float = 0.01
