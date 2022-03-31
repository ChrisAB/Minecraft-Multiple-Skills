import gym
import sys
import minerl
import argparse
import time
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import re
import numpy as np
import logging
import copy
import cv2
import time
from pathlib import Path
from PIL import Image
from collections import OrderedDict

import missions
from utils.action_converter import ActionConverter
from image_processors.basic_processor import BasicImageProcessor
from neural_networks.dqn import DQN
from utils.replay_memory import Transition, ReplayMemory
from itertools import count
from deep_skill_module.dsn_array_module import DSNArrayModule
from deep_skill_module.dsn import DSNModule
from neural_networks.q_network import QNetwork
from control_blocks.basic_control_block import ControlBlock
from configs import train_individual_skills_config
logging.basicConfig(level=logging.ERROR)


class DSNArrayModuleSkillTrainer:
    def __init__(self, args):
        self.loaded = False
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.initenv()

        self.init_network_to_be_trained()

        self.init_optimizer()
        self.init_replay_memory()
        self.init_image_processor()
        self.init_hyper_parameters()
        self.init_episode()

    # Load the DSNs which we train from

    def load_dsn_array(self):
        self.array_of_DSNs = []
        self.DSN_saves = []

        for DSN_location in self.args.ARRAY_OF_DSN_LOCATIONS:
            current_save = torch.load(DSN_location)
            self.array_of_DSNs.append(current_save.module)
            # if(current_save.nn_type == 'DQN'):

            #     dqn = DQN(current_save.input_dimensions, current_save.inputs, current_save.filters,
            #               current_save.strides, current_save.number_of_hidden_neurons, current_save.number_of_actions, current_save.nl)
            #     self.array_of_DSN.append(DQN(current_save.nn))
            # if(current_save.nn_type == 'DSNArrayModule'):
            #     self.array_of_DSN.append(
            #         DSNArrayModule(current_save.nn, QNetwork(), ControlBlock()))
            self.DSN_saves.append(current_save)

    def load_q_network(self):
        if(self.loaded and self.loaded.q_network):
            self.q_network = self.loaded.q_network
        else:
            print("Could not find q_network saved in checkpoint. Using a new basic one")
            self.q_network = QNetwork(
                self.args.INPUT_DIMENSIONS[0]*self.args.INPUT_DIMENSIONS[1], self.args.INPUT_DIMENSIONS[2])

    def load_control_block(self):
        if(self.loaded and self.loaded.control_block):
            self.control_block = self.loaded.control_block
        else:
            print(
                "Could not find control_block saved in checkpoint. Using a new basic one")
            self.control_block = ControlBlock()

    def load_dsn_module(self):
        if(self.loaded and self.loaded.module):
            self.dsn = self.loaded.module
        else:
            print("Could not find module saved in checkpoint. Using a new dsn array module initialized with array of DSNs")
            self.dsn = DSNArrayModule(self.array_of_DSNs)

    def load_checkpoint(self):
        try:
            self.loaded = torch.load(self.args.CHECKPOINT_SAVE_LOCATION +
                                     self.args.mission + '_' + str(self.args.episode) + '.pt')
        except:
            print("Could not load checkpoint from location: " + self.args.CHECKPOINT_SAVE_LOCATION +
                  self.args.mission + '_' + str(self.args.episode) + '.pt')
            print("Exiting...")
            sys.exit(1)

    def initenv(self):
        self.env = gym.make(self.args.mission)
        self.action_converter = ActionConverter(self.env.action_space.noop())
        self.n_actions = len(self.action_converter.actions_array)

    def init_network_to_be_trained(self):

        self.load_dsn_array()

        if(self.args.episode != 0):
            self.load_checkpoint()

        self.load_q_network()
        self.load_control_block()
        self.load_dsn_module()

        self.dsn = DSNModule(self.dsn, self.q_network, self.control_block)

    def init_optimizer(self):
        self.optimizer = torch.optim.RMSprop(self.dsn.parameters())

    def init_replay_memory(self):
        self.memory = ReplayMemory(self.args.REPLAY_MEMORY_SIZE)

    def init_image_processor(self):
        self.image_processor = BasicImageProcessor(self.args)

    def init_hyper_parameters(self):
        self.BATCH_SIZE = self.args.BATCH_SIZE
        self.GAMMA = self.args.GAMMA
        self.EPS_START = self.args.EPS_START
        self.EPS_END = self.args.EPS_END
        self.EPS_DECAY = self.args.EPS_DECAY
        self.TARGET_UPDATE = self.args.TARGET_UPDATE

    def init_episode(self):
        if(self.loaded):
            self.steps_done = self.loaded.steps_done
        else:
            self.steps_done = 0
        if(self.loaded):
            self.episode_durations = self.loaded.episode_durations
        else:
            self.episode_durations = []
        self.current_episode_observation = []
        self.current_episode_actions = []
        self.current_episode_rewards = []

    def calculate_eps_threshold(self):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)

    def get_prediction(self, state):
        return self.dsn(state)

    def select_action_using_epsilon_greedy(self, state):
        sample = random.random()
        eps_threshold = self.calculate_eps_threshold()

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                prediction = self.get_prediction(state)
                best_prediction = prediction.max(1)[1]
                return self.action_converter.get_action(best_prediction)
        else:
            return self.env.action_space.sample()

    def select_action(self, state):
        return self.select_action_using_epsilon_greedy(state)

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
        obs = self.image_processor.transform(obs)
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

    def reset_env(self):
        self.env_reset_start_time = time.time()

        self.raw_obs = self.env.reset()
        self.env.render(mode="rgb_array")

        print("Env was reset. Took {} seconds to reset it.".format(
            time.time() - self.env_reset_start_time), flush=True)

    def restart_episode_start_time(self):
        self.episode_start_time = time.time()

    def reset_episode_metrics(self):
        self.restart_episode_start_time()

        self.current_episode_actions = []
        self.current_episode_observation = []
        self.current_episode_rewards = []

    def start_episode(self, episode):
        print("Starting episode {}. Last episode took {} seconds".format(
            episode, time.time()-self.episode_start_time), flush=True)

        self.reset_env()
        self.reset_episode_metrics()

    def save_episode(self, episode):
        if self.args.savevideosteps > 0 and episode % self.args.savevideosteps == 0 and episode != self.args.episode:
            out = cv2.VideoWriter(
                self.args.REPLAY_CAPTURE_LOCATION + 'gif_' + str(episode) + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (self.args.INPUT_DIMENSIONS[1], self.args.INPUT_DIMENSIONS[2]))
            for i in self.current_episode_observation:
                out.write(i)
            out.release()

    def store_current_state_in_replay_memory(self):
        self.action = np.array(self.action)
        self.action = torch.from_numpy(self.action).to(self.device)
        self.memory.push(self.obs, self.action, self.next_obs, self.reward)

    def save_current_image(self, episode, frame):
        if self.args.saveimagesteps > 0 and frame % self.args.saveimagesteps == 0 and frame != 0:
            print('Saved image ' + str(episode) +
                  '_' + str(t) + '.png', flush=True)
            img_obs = self.obs['pov']
            img_obs = np.flipud(img_obs)
            img = Image.fromarray(img_obs)
            img.save(self.args.IMAGE_CAPTURE_LOCATION + 'image' +
                     str(episode) + '_' + str(frame) + '.png')

    def save_checkpoint(self, episode):
        torch.save({'steps_done': self.steps_done,
                    'episode_durations': self.episode_durations,
                    'nn': self.policy_net.state_dict()},
                   self.args.CHECKPOINT_SAVE_LOCATION + self.args.mission + '_' + str(i_episode) + '.pt')

    def end_of_episode_processing(self, episode):
        self.plot_durations()

    def play_episode(self, episode):
        self.obs = self.preprocess_image(self.raw_obs)

        for frame in count():
            # Select and perform an action
            self.action = self.select_action(self.obs)
            self.raw_obs, self.reward, done, _ = self.env.step(self.action)

            self.current_episode_rewards += [self.reward]
            self.reward = torch.tensor([self.reward], device=self.device)

            if not done:
                self.next_obs = self.preprocess_image(self.raw_obs)
            else:
                self.next_obs = None
                self.save_episode(episode)

            self.action = self.action_converter.convert_to_array(self.action)
            self.current_episode_actions += [self.action]

            self.store_current_state_in_replay_memory()

            self.optimize_model()

            if done:
                self.episode_durations.append(frame + 1)
                break

            self.save_current_image(episode, frame)

        print("Finished episode", flush=True)

        self.save_checkpoint(episode)
        self.end_of_episode_processing(episode)

    def start(self):
        print("Starting...", flush=True)
        self.episode_start_time = time.time()

        for i_episode in range(self.args.episode, self.args.episodes):

            # Initialize the environment and state
            self.start_episode(i_episode)
            self.play_episode(i_episode)

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
    parser.add_argument('--ARRAY_OF_DSN_LOCATIONS', type=list,
                        default=train_individual_skills_config.ARRAY_OF_DSN_LOCATIONS, help='location of DSNs from which to learn from')
    args = parser.parse_args()

    if args.server2 is None:
        args.server2 = args.server

    trainer = DSNArrayModuleSkillTrainer(args)
    trainer.start()
