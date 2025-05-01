import json
import torch
from src.environment import CryptoTradingEnvironment, Balance
from src.agent import Agent, LoadedAgent, BaseAgent
from src.train_config import QTradingConfigurations
from src import train_utils


def run_single_episode(configs: QTradingConfigurations, env: CryptoTradingEnvironment, agent: BaseAgent, render: bool = False):
    state, info = env.reset()
    state = state.unsqueeze(0)
    while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        if render:
            env.render()

        state = None if terminated else observation.unsqueeze(0)
        if done:
            break

    return reward


def main():
    with open(r"configurations/config.json") as f:
        config_data = json.load(f)
    configs = QTradingConfigurations(**config_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CryptoTradingEnvironment(configs.env_parameters, device, configs.model_dir)
    n_actions = env.action_space.n
    if configs.mode == "use":
        agent = LoadedAgent(env.observation_shape, n_actions, configs, device)
        run_single_episode(configs, env, agent, render=True)
    else:
        agent = Agent(env.observation_shape, n_actions, configs, device)
        train_utils.train(configs, agent, env)


if __name__ == "__main__":
    main()

