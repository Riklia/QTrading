from src.agent.base_agent import BaseAgent
from src.environment import ObservationShape
from src.train_config import QTradingConfigurations


class LoadedAgent(BaseAgent):
    def __init__(self, observation_shape: ObservationShape, n_actions: int, configs: QTradingConfigurations):
        super().__init__(observation_shape, n_actions, configs)
        self.load_model(self.configs.model_dir, self.configs.model_name, self.device)
