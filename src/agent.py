import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import QNetwork
from src.replay_buffer import ReplayMemory, Transition
from src.train_config import TrainConfig


class Agent:
    def __init__(self, n_observations: int, n_actions: int, configs: TrainConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(n_observations, n_actions).to(self.device)
        self.target_net = QNetwork(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=configs.learning_parameters.lr, amsgrad=True)
        self.recurrent_cell = self.policy_net.init_recurrent_cell_states(1, self.device)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.configs = configs
        self.replay_memory = ReplayMemory(10000)
        self.episode_usd_final_balance = []
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
        hx_batch = torch.cat(batch.hx)
        cx_batch = torch.cat(batch.cx)
        recurrent_cell_batch = (hx_batch.unsqueeze(0), cx_batch.unsqueeze(0))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values, _ = self.policy_net(state_batch, recurrent_cell_batch, len(state_batch))
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)

        target_hx_batch = hx_batch.clone()
        target_cx_batch = cx_batch.clone()
        target_hx_batch = target_hx_batch[non_final_mask]
        target_cx_batch = target_cx_batch[non_final_mask]
        target_recurrent_cell_batch = (target_hx_batch.unsqueeze(0), target_cx_batch.unsqueeze(0))
        with torch.no_grad():
            out, _ = self.target_net(non_final_next_states, target_recurrent_cell_batch, len(non_final_next_states))
            next_state_values[non_final_mask] = out.max(1).values
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
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                out, self.recurrent_cell = self.policy_net(state, self.recurrent_cell)
                return out.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_durations(self, array_to_plot=None, plot_name: str = "learning_plot.png",
                       y_label: str = "Final USD balance", show_result=False):

        if array_to_plot is None:
            array_to_plot = torch.tensor(self.episode_usd_final_balance, dtype=torch.float)
        else:
            array_to_plot = torch.tensor(array_to_plot, dtype=torch.float)
        save_every = self.configs.learning_parameters.save_plot_every
        episodes_so_far = len(array_to_plot)

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
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        if save_every != 0 and episodes_so_far % save_every == 0:
            plt.savefig(f"{self.configs.model_dir}/{plot_name}")
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
