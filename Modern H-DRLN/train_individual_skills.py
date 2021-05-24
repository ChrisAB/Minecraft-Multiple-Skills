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
from collections import OrderedDict
from utils.action_converter import ActionConverter
from configs import train_individual_skills_config
import logging
import missions
import copy
import cv2
import time
logging.basicConfig(level=logging.ERROR)


class SkillTrainer:
    def __init__(self, args):
        self.loaded = False
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.initenv()
        self.policy_net = DQN(
            input_dimensions=self.args.INPUT_DIMENSIONS, number_of_actions=self.n_actions).to(self.device)
        self.target_net = DQN(
            input_dimensions=self.args.INPUT_DIMENSIONS, number_of_actions=self.n_actions).to(self.device)
        if(self.args.episode != 0):
            self.loaded = torch.load(
                self.args.CHECKPOINT_SAVE_LOCATION + self.args.mission + '_' + str(self.args.episode) + '.pt')
            try:
                self.policy_net.load_state_dict(self.loaded.nn)
                self.policy_net.eval()
            except:
                print("Could not load checkpoint from location: " + self.args.CHECKPOINT_SAVE_LOCATION +
                      self.args.mission + '_' + str(self.args.episode) + '.pt')
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.args.REPLAY_MEMORY_SIZE)
        self.grayscaler = torchvision.transforms.Grayscale()

        self.BATCH_SIZE = self.args.BATCH_SIZE
        self.GAMMA = self.args.GAMMA
        self.EPS_START = self.args.EPS_START
        self.EPS_END = self.args.EPS_END
        self.EPS_DECAY = self.args.EPS_DECAY
        self.TARGET_UPDATE = self.args.TARGET_UPDATE

        self.steps_done = 0
        if(self.loaded):
            self.steps_done = self.loaded.steps_done
        self.episode_durations = []
        if(self.loaded):
            self.episode_durations = self.loaded.episode_durations
        self.current_episode_observation = []
        self.current_episode_actions = []
        self.current_episode_rewards = []

    def initenv(self):
        self.env = gym.make(self.args.mission)
        self.action_converter = ActionConverter(self.env.action_space.noop())
        self.n_actions = len(self.action_converter.actions_array)

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
                actions = copy.deepcopy(self.action_converter.actions_array)
                actions[action] = 1
                return self.action_converter.convert_to_ordereddict(actions)
        else:
            return self.env.action_space.sample()

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
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch)
        state_action_values = state_action_values.gather(0, action_batch)

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
        obs = obs['pov']
        self.current_episode_observation += [obs]
        obs = torchvision.transforms.functional.to_tensor(
            np.flip(obs, axis=0).copy())
        obs = obs.reshape(1, 3, 64, 64)
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

        plt.savefig(self.args.PLOT_IMAGE_LOCATION, bbox_inches='tight')

    def start(self):
        print("Starting...", flush=True)
        start_time = time.time()
        for i_episode in range(self.args.episode, self.args.episodes):
            print("Starting episode {}. Last episode took {} seconds".format(
                i_episode, time.time()-start_time), flush=True)
            start_time = time.time()
            # Initialize the environment and state
            obs = self.env.reset()
            self.current_episode_observation = []
            self.current_episode_actions = []
            self.current_episode_rewards = []
            self.env.render(mode="rgb_array")
            print("Env was reset. Took {} seconds to reset it.".format(
                time.time() - start_time), flush=True)
            state = self.preprocess_image(obs)
            print("Image was preprocessed", flush=True)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                obs, reward, done, info = self.env.step(action)
                self.current_episode_rewards += reward
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if not done:
                    next_state = self.preprocess_image(obs)
                else:
                    next_state = None
                    if self.args.savevideosteps > 0 and i_episode % self.args.savevideosteps == 0 and i_episode != self.args.episode:
                        out = cv2.VideoWriter(
                            self.args.REPLAY_CAPTURE_LOCATION + 'gif_' + str(i_episode) + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (self.args.INPUT_DIMENSIONS[1], self.args.INPUT_DIMENSIONS[2]))
                        for i in self.current_episode_observation:
                            out.write(i)
                        out.release()

                # Store the transition in memory
                action = self.action_converter.convert_to_array(action)
                self.current_episode_actions += [action]
                action = np.array(action)
                action = torch.from_numpy(action).to(self.device)
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    break
                if self.args.saveimagesteps > 0 and t % self.args.saveimagesteps == 0 and t != 0:
                    print('Saved image ' + str(i_episode) +
                          '_' + str(t) + '.png', flush=True)
                    img_obs = obs['pov']
                    img_obs = np.flipud(img_obs)
                    img = Image.fromarray(img_obs)
                    img.save(self.args.IMAGE_CAPTURE_LOCATION + 'image' +
                             str(i_episode) + '_' + str(t) + '.png')
            print("Finished episode", flush=True)
            torch.save({
                'observations': self.current_episode_observation,
                'actions': self.current_episode_actions,
                'rewards': self.current_episode_rewards
            }, './data_out' + self.args.mission + '_' + str(i_episode) + '.pt')
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                print("Updating target at episode " +
                      str(i_episode), flush=True)
                torch.save({'steps_done': self.steps_done,
                            'episode_durations': self.episode_durations,
                            'memory': self.memory,
                            'nn': self.policy_net.state_dict()},
                           self.args.CHECKPOINT_SAVE_LOCATION + self.args.mission + '_' + str(i_episode) + '.pt')
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.plot_durations()

        print('Complete', flush=True)
        self.env.render()
        self.env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MALMOENV')
    parser.add_argument('--port', type=int, default=train_individual_skills_config.PORT,
                        help='the mission server port')
    parser.add_argument('--server', type=str, default=train_individual_skills_config.SERVER,
                        help='the mission server DNS or IP address')
    parser.add_argument('--mission', type=str, default=train_individual_skills_config.MISSION,
                        help='the mission ID')
    parser.add_argument('--port2', type=int, default=train_individual_skills_config.PORT2,
                        help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=train_individual_skills_config.SERVER2,
                        help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=train_individual_skills_config.EPISODES,
                        help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=train_individual_skills_config.EPISODE,
                        help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=train_individual_skills_config.ROLE,
                        help='the agent role - defaults to 0')
    parser.add_argument('--saveimagesteps', type=int,
                        default=train_individual_skills_config.SAVEIMAGESTEPS, help='save an image every N steps')
    parser.add_argument('--savevideosteps', type=int,
                        default=train_individual_skills_config.SAVEVIDEOSTEPS, help='save a video every N episodes')
    parser.add_argument('--BATCH_SIZE', type=int,
                        default=train_individual_skills_config.BATCH_SIZE, help='batch size')
    parser.add_argument('--GAMMA', type=float, default=train_individual_skills_config.GAMMA,
                        help='Gamma for expected Q values')
    parser.add_argument('--EPS_START', type=float,
                        default=train_individual_skills_config.EPS_START, help='epsilon start')
    parser.add_argument('--EPS_END', type=float,
                        default=train_individual_skills_config.EPS_END, help='epsilon end')
    parser.add_argument('--EPS_DECAY', type=float,
                        default=train_individual_skills_config.EPS_DECAY, help='epsilon decay')
    parser.add_argument('--INPUT_DIMENSIONS', type=list,
                        default=train_individual_skills_config.INPUT_DIMENSIONS, help='DQN input dimensions')
    parser.add_argument('--TARGET_UPDATE', type=int, default=train_individual_skills_config.TARGET_UPDATE,
                        help='how often to update target_dict. Also how often to save checkpoint')
    parser.add_argument('--REPLAY_MEMORY_SIZE', type=int,
                        default=train_individual_skills_config.REPLAY_MEMORY_SIZE, help='replay memory size')
    parser.add_argument('--IMAGE_CAPTURE_LOCATION', type=str,
                        default=train_individual_skills_config.IMAGE_CAPTURE_LOCATION, help='where to capture images during episodes')
    parser.add_argument('--REPLAY_CAPTURE_LOCATION', type=str,
                        default=train_individual_skills_config.REPLAY_CAPTURE_LOCATION, help='where to capture the replay')
    parser.add_argument('--CHECKPOINT_SAVE_LOCATION', type=str,
                        default=train_individual_skills_config.CHECKPOINT_SAVE_LOCATION, help='where to save checkpoints')
    parser.add_argument('--PLOT_IMAGE_LOCATION', type=str, default=train_individual_skills_config.PLOT_IMAGE_LOCATION,
                        help='where to plot the graph showing duration/episode')
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server
    trainer = SkillTrainer(args)
    trainer.start()
