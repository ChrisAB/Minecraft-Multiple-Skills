from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import sys
from typing import List
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers

OBTAIN_STEPS = 2000


class ObtainLog(SimpleEmbodimentEnvSpec):
    def __init__(self, dense=False, extreme=False, *args, **kwargs):
        name = 'MineRLObtainLog-v0'
        self.dense = dense
        self.extreme = extreme
        super().__init__(name, *args, max_episode_steps=OBTAIN_STEPS,
                         reward_threshold=5.0, **kwargs)

    def is_from_folder(self, folder: str) -> bool:
        return 'obtain'

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            handlers.CompassObservation(angle=True, distance=False),
            handlers.FlatInventoryObservation(['dirt'])]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables()

    # john rl nyu microsfot van roy and ian osband

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
            ])
        ]

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type="iron_axe", quantity=1)
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type='log', amount=5)
            ])
        ]

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True
            )
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(OBTAIN_STEPS * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
            min_randomized_distance=0,
            randomize_compass_location=True
        ), handlers.NavigationDecorator(
            max_randomized_radius=1,
            min_randomized_radius=1,
            block='log',
            placement='surface',
            max_radius=1,
            min_radius=0,
            max_randomized_distance=1,
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
        return "Simple Obtain 5 logs spawned around the agent"

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold
