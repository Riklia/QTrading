import torch
from src.model import QNetwork
from src.replay_buffer import ReplayMemory
from src.train_config import QTradingConfigurations
from src.environment import ObservationShape


class BaseAgent:
    def __init__(self, observation_shape: ObservationShape, n_actions: int, configs: QTradingConfigurations):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(observation_shape, n_actions).to(self.device)
        self.target_net = QNetwork(observation_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.configs = configs
        self.replay_memory = ReplayMemory(20000)

    def select_action(self, state):
        with torch.no_grad():
            out = self.policy_net(state)
            return out.max(1).indices.view(1, 1)

    def save_model(self, directory, filename):
        torch.save(self.policy_net.state_dict(), f"{directory}/{filename}_policy.pth")
        torch.save(self.target_net.state_dict(), f"{directory}/{filename}_target.pth")

    def load_model(self, directory, model_name, device: torch.device):
        self.policy_net.load_state_dict(torch.load(f"{directory}/{model_name}_policy.pth", map_location=device))
        self.target_net.load_state_dict(torch.load(f"{directory}/{model_name}_target.pth", map_location=device))
