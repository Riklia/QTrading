import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.replay_buffer import Transition
from src.train_config import QTradingConfigurations
from src.environment import ObservationShape
from src.agent.base_agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, observation_shape: ObservationShape, n_actions: int, configs: QTradingConfigurations, device: torch.device):
        super().__init__(observation_shape, n_actions, configs, device)
        if configs.learning_parameters.continue_learning:
            self.load_model(configs.learning_parameters.load_model_directory, configs.model_name, self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=configs.learning_parameters.lr, amsgrad=True)
        self.episode_rewards = []
        self.steps_done = 0

    def learn(self):
        batch_size = self.configs.learning_parameters.batch_size
        gamma = self.configs.learning_parameters.gamma
        if len(self.replay_memory) < batch_size:
            return
        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # action_probs = nn.functional.softmax(state_action_values, dim=1)
        # entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-9), dim=-1).mean()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            best_next_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)[1]
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,
                                                                                              best_next_actions).squeeze()

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).float())
        # print("Loss:", loss.item())
        # loss -= 1e-3 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state):
        eps_end = self.configs.learning_parameters.eps_end
        eps_start = self.configs.learning_parameters.eps_start
        eps_decay = self.configs.learning_parameters.eps_decay
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                out = self.policy_net(state)
                # print("Q-values:", out.cpu().detach().numpy())
                return out.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self, array_to_plot, plot_name: str = "learning_plot.png",
                       y_label: str = "Final USD balance", show_result=False, save_model=False):
        episodes_so_far = len(array_to_plot)
        save_every = self.configs.learning_parameters.save_plot_every
        if save_every == 0 or episodes_so_far % save_every != 0:
            return
        
        array_to_plot = torch.tensor(array_to_plot, dtype=torch.float)

        plt.figure(1)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel(y_label)
        plt.plot(array_to_plot.numpy())
        # Plot average of 100 last episodes
        if episodes_so_far >= 100:
            means = array_to_plot.unfold(0, 100, 1).mean(1).view(-1)
            plt.plot(torch.arange(99, means.size(0) + 99), means.numpy())
        plt.savefig(f"{self.configs.model_dir}/{plot_name}")
        if save_model:
            self.save_model(self.configs.model_dir, self.configs.model_name)


    def save_plot(self, directory, filename):
        plt.figure(1)
        final_rewards = torch.tensor(self.episode_rewards, dtype=torch.float)
        plt.title("Result")
        plt.xlabel("Episode")
        plt.ylabel("Final cumulative reward")
        plt.plot(final_rewards.numpy())
        if len(final_rewards) >= 100:
            means = final_rewards.unfold(0, 100, 1).mean(1).view(-1)
            plt.plot(torch.arange(99, means.size(0) + 99), means.numpy())
        
        plt.savefig(f"{directory}/{filename}.png")

    def save_model(self, directory, model_name):
        torch.save(self.policy_net.state_dict(), f"{directory}/{model_name}_policy.pth")
        torch.save(self.target_net.state_dict(), f"{directory}/{model_name}_target.pth")
        torch.save(self.optimizer.state_dict(), f"{directory}/{model_name}_optimizer.pth")

    def load_model(self, directory, model_name, device: torch.device):
        self.policy_net.load_state_dict(torch.load(f"{directory}/{model_name}_policy.pth", map_location=device))
        self.target_net.load_state_dict(torch.load(f"{directory}/{model_name}_target.pth", map_location=device))
        self.optimizer.load_state_dict(torch.load(f"{directory}/{model_name}_optimizer.pth", map_location=device))

    def __del__(self):
        self.save_model(self.configs.model_dir, self.configs.model_name)
        self.save_plot(self.configs.model_dir, self.configs.model_name)
