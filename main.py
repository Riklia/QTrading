import json
import torch
from src.environment import CryptoTradingEnvironment, Balance
from src.agent import Agent, LoadedAgent, BaseAgent
from src.train_config import QTradingConfigurations
from src import train_utils


def run_single_episode(configs: QTradingConfigurations, env: CryptoTradingEnvironment, agent: BaseAgent, render: bool = False):
    state, info = env.reset()
    state = torch.tensor(state, device=agent.device).unsqueeze(0)
    while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        if render:
            env.render(configs.model_dir)

        state = None if terminated else torch.tensor(observation, device=agent.device).unsqueeze(0)
        if done:
            break

    return env.get_overall_current_balance() - env.initial_overall_balance


def main():
    with open(r"configurations/config.json") as f:
        config_data = json.load(f)
    configs = QTradingConfigurations(**config_data)
    balance_log_path = None
    if configs.env_parameters.record_balance:
        balance_log_path = f"{configs.model_dir}/balance_log.txt"

    initial_balance = Balance(balance_log_path)
    initial_balance.update_balance("USD", configs.env_parameters.initial_usd_balance)
    initial_balance.update_balance("BTC", configs.env_parameters.initial_btc_balance)
    env = CryptoTradingEnvironment(initial_balance, configs.env_parameters)
    n_actions = env.action_space.n
    if configs.mode == "use":
        agent = LoadedAgent(env.observation_shape, n_actions, configs)
        run_single_episode(configs, env, agent, render=True)
    else:
        agent = Agent(env.observation_shape, n_actions, configs)
        train_utils.train(configs, agent, env)


if __name__ == "__main__":
    main()

