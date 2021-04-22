import gym
import minerl
import argparse
from pathlib import Path
import time
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from dqn import DQN
from replay_memory import Transition, ReplayMemory
from itertools import count
import random
import math
import re
import numpy as np
from array2gif import write_gif


class SkillTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.initenv()

        self.policy_net = DQN(number_of_actions=self.n_actions).to(self.device)
        self.target_net = DQN(number_of_actions=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.grayscaler = torchvision.transforms.Grayscale()

        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.steps_done = 0
        self.episode_durations = []

    def initenv(self):
        self.env = gym.make('MineRLNavigateDense-v0')

        self.n_actions = self.env.action_space.__len__()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(state).max(1)[1].view(1, 1)
                action = action.item()
                action = self.env.action_space.__getitem__(action)
                action = int(re.search(r'\d+', action).group())
                return torch.tensor([[action]], device=self.device, dtype=torch.long)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values,
                                                  expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def preprocess_image(self, obs):
        obs = obs.reshape(84, 84, 3)
        np.flipud(obs)
        obs = torchvision.transforms.functional.to_tensor(obs)
        obs = self.grayscaler(obs)
        obs = obs.reshape(1, 1, 84, 84)
        return obs.to(self.device)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        print(self.episode_durations, flush=True)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.savefig(self.args.plot_image_location, bbox_inches='tight')

    def start(self):
        print("Starting...", flush=True)
        num_episodes = self.args.episodes
        for i_episode in range(num_episodes):
            print("Starting episode " + str(i_episode), flush=True)
            # Initialize the environment and state
            obs = self.env.reset()
            print("Env was reset", flush=True)
            state = self.preprocess_image(obs)
            print("Image was preprocessed", flush=True)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                obs, reward, done, info = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if not done:
                    next_state = self.preprocess_image(obs)
                else:
                    next_state = None
                    write_gif(self.memory.memory[self.memory.position-t:self.memory.position],
                              'tmp_images/gif_' + str(i_episode) + '.gif', fps=5)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    break
                if self.args.saveimagesteps > 0 and t % self.args.saveimagesteps == 0:
                    print('Saved image ' + str(i_episode) +
                          '_' + str(t) + '.png', flush=True)
                    img_obs = obs.copy()
                    img_obs = img_obs.reshape(84, 84, 3)
                    img_obs = np.flipud(img_obs)
                    img = Image.fromarray(img_obs)
                    img.save('tmp_images/image' +
                             str(i_episode) + '_' + str(t) + '.png')
            print("Finished episode", flush=True)
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                print("Updating target at episode " +
                      str(i_episode), flush=True)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.plot_durations()

        print('Complete', flush=True)
        self.env.render()
        self.env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MALMOENV')
    parser.add_argument('--port', type=int, default=10000,
                        help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1',
                        help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None,
                        help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None,
                        help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1,
                        help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0,
                        help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0,
                        help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int,
                        default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int,
                        default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str,
                        default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server
    args.plot_image_location = '/content/plot.png'
    trainer = SkillTrainer(args)
    trainer.start()
