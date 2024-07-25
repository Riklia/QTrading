import matplotlib.pyplot as plt
import torch
from src.environment import CryptoTradingEnvironment
from src.agent import Agent
from src.train_config import QTradingConfigurations


def run_training_episode(configs: QTradingConfigurations, env: CryptoTradingEnvironment, agent: Agent, render: bool = False, update_target_every=100):
    state, info = env.reset()
    state = torch.tensor(state, device=agent.device).unsqueeze(0)
    tau = configs.learning_parameters.tau
    step_count = 0

    while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=agent.device)
        done = terminated or truncated
        if render:
            env.render(configs.model_dir)

        next_state = None if terminated else torch.tensor(observation, device=agent.device).unsqueeze(0)
        agent.replay_memory.push(state, action, next_state, reward)
        state = next_state
        agent.learn()

        if step_count % update_target_every == 0:
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            agent.target_net.load_state_dict(target_net_state_dict)

        step_count += 1

        if done:
            agent.episode_usd_final_balance.append(env.get_overall_current_balance())
            agent.plot_durations()
            agent.episode_rewards.append(reward)
            agent.plot_durations(array_to_plot=agent.episode_rewards, plot_name="training_rewards.png", y_label="Final reward")
            break

    return env.get_overall_current_balance() - env.initial_overall_balance


def train(configs: QTradingConfigurations, agent, env):
    plt.ion()
    num_episodes = configs.learning_parameters.num_episodes
    print_frequency = configs.learning_parameters.print_frequency
    for i_episode in range(num_episodes):
        episode_profit = run_training_episode(configs, env, agent, render=configs.learning_parameters.render)
        if i_episode % print_frequency == 0 and i_episode > 0:
            print(f"Episode: {i_episode:5}  Profit: {episode_profit:5}")

    print("Complete")
    agent.plot_durations(show_result=True)
    env.render(configs.model_dir)
    plt.ioff()
    plt.show()

