import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from environment import CryptoTradingEnvironment, Balance
from agent import Agent
from configurations.config import TrainConfig


def run_episode(configs: TrainConfig, env: CryptoTradingEnvironment, agent: Agent):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    render = configs.learning_parameters.render
    tau = configs.learning_parameters.tau
    while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=agent.device)
        done = terminated or truncated
        if render:
            env.render(configs.model_dir)
        # print("time_point: ", env.time_point)
        # print("reward: ", reward)
        # print("overall_current_balance: ", env._get_overall_current_balance())
        # print("overall_usd_balance: ", env.current_balance["USD"])
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

        agent.replay_memory.push(state, action, next_state, reward)
        state = next_state
        agent.learn()

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        agent.target_net.load_state_dict(target_net_state_dict)

        if done:
            agent.episode_usd_final_balance.append(env.get_overall_current_balance())
            agent.plot_durations()
            break

    return env.get_overall_current_balance() - env.initial_overall_balance


def train(configs: TrainConfig, agent, env):
    plt.ion()
    num_episodes = configs.learning_parameters.num_episodes
    print_frequency = configs.learning_parameters.print_frequency
    for i_episode in range(num_episodes):
        episode_profit = run_episode(configs, env, agent)
        if i_episode % print_frequency == 0 and i_episode > 0:
            print(f"Episode: {i_episode:5}  Profit: {episode_profit:5}")

    print("Complete")
    agent.plot_durations(show_result=True)
    env.render(configs.model_dir)
    plt.ioff()
    plt.show()


def main():
    with open(r"configurations/config.json") as f:
        config_data = json.load(f)
    configs = TrainConfig(**config_data)
    balance_log_path = None
    if configs.env_parameters.record_balance:
        balance_log_path = f"{configs.model_dir}/balance_log.txt"

    initial_balance = Balance(balance_log_path)
    initial_balance.update_balance("USD", configs.env_parameters.initial_balance)
    initial_balance.register_currency("BTC")
    env = CryptoTradingEnvironment(initial_balance, configs.env_parameters)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    agent = Agent(n_observations, n_actions, configs)

    train(configs, agent, env)
    # it is in the destructor now
    # agent.save_model(model_dir, model_name)


if __name__ == "__main__":
    main()

