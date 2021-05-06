from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import sys
from typing import List
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers

NAVIGATE_STEPS = 6000


class NavigateTree(SimpleEmbodimentEnvSpec):
    def __init__(self, dense=False, extreme=False, *args, **kwargs):
        name = 'MineRLNavigateTree-v0'
        self.dense = dense
        self.extreme = extreme
        super().__init__(name, *args, max_episode_steps=6000, **kwargs)

    def is_from_folder(self, folder: str) -> bool:
        return 'navigate'

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            handlers.CompassObservation(angle=True, distance=False),
            handlers.FlatInventoryObservation(['dirt'])]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            handlers.PlaceBlock(['none', 'dirt'],
                                _other='none', _default='none')]

    # john rl nyu microsfot van roy and ian osband

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForTouchingBlockType([
                {'type': 'log', 'behaviour': 'onceOnly',
                 'reward': 100.0},
            ])
        ] + ([handlers.RewardForDistanceTraveledToCompassTarget(
            reward_per_block=1.0
        )] if self.dense else [])

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type='compass', quantity='1')
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromTouchingBlockType(
                ["log"]
            )
        ]

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True
            )
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(NAVIGATE_STEPS * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [handlers.NavigationDecorator(
            max_randomized_radius=64,
            min_randomized_radius=64,
            block='log',
            placement='surface',
            max_radius=8,
            min_radius=0,
            max_randomized_distance=8,
            min_randomized_distance=0,
            randomize_compass_location=True
        )]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000
            ),
            handlers.WeatherInitialCondition('clear'),
            handlers.SpawningInitialCondition('false')
        ]

    def get_docstring(self):
        return "Simple Find the Tree"

    def determine_success_from_rewards(self, rewards: list) -> bool:
        reward_threshold = 100.0
        return sum(rewards) >= reward_threshold
