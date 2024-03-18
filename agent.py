import math
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
from replay_buffer import ReplayMemory, Transition
from configurations.config import TrainConfig

steps_done = 0


class Agent:
    def __init__(self, n_observations: int, n_actions: int, configs: TrainConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(n_observations, n_actions).to(self.device)
        self.target_net = QNetwork(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=configs.learning_parameters.lr, amsgrad=True)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.configs = configs
        self.replay_memory = ReplayMemory(10000)
        self.episode_usd_final_balance = []

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state):
        eps_end = self.configs.learning_parameters.eps_end
        eps_start = self.configs.learning_parameters.eps_start
        eps_decay = self.configs.learning_parameters.eps_decay
        global steps_done
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self, show_result=False):
        final_balances = torch.tensor(self.episode_usd_final_balance, dtype=torch.float)

        save_every = self.configs.learning_parameters.save_plot_every
        episodes_so_far = len(final_balances)

        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Final USD balance')
        plt.plot(final_balances.numpy())
        # Plot average of 100 last episodes
        if episodes_so_far >= 100:
            means = final_balances.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        if save_every != 0 and episodes_so_far % save_every == 0:
            plt.savefig(f"{self.configs.model_dir}/learning_plot.png")
        plt.pause(0.001)

    def save_plot(self, directory, filename):
        plt.figure(1)
        final_balances = torch.tensor(self.episode_usd_final_balance, dtype=torch.float)
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Final USD balance')
        plt.plot(final_balances.numpy())
        if len(final_balances) >= 100:
            means = final_balances.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.savefig(f"{directory}/{filename}.png")

    def save_model(self, directory, filename):
        torch.save(self.policy_net.state_dict(), f"{directory}/{filename}_policy.pth")
        torch.save(self.target_net.state_dict(), f"{directory}/{filename}_target.pth")

    def __del__(self):
        self.save_model(self.configs.model_dir, self.configs.model_name)
        self.save_plot(self.configs.model_dir, self.configs.model_name)
