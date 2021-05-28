from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import sys
from typing import List
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers

CRAFT_STEPS = 1000


class CraftWoodenPickaxe(SimpleEmbodimentEnvSpec):
    def __init__(self, dense=False, extreme=False, *args, **kwargs):
        name = 'MineRLCraftWoodenPickaxe-v0'
        self.dense = dense
        self.extreme = extreme
        self.reward_schedule = [
            dict(type="planks", amount=1, reward=2),
            dict(type="stick", amount=1, reward=4),
            dict(type="crafting_table", amount=1, reward=4),
            dict(type="wooden_pickaxe", amount=1, reward=8),
        ]
        super().__init__(name, *args, max_episode_steps=CRAFT_STEPS, **kwargs)

    def is_from_folder(self, folder: str) -> bool:
        return 'navigate'

    def create_observables(self) -> List[Handler]:
        return super().create_observables()

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            handlers.PlaceBlock(['none', 'crafting_table'],
                                _other='none', _default='none'),
            handlers.CraftAction(
                ['none', 'stick', 'planks', 'crafting_table'], _other='none', _default='none'),
            handlers.CraftNearbyAction(
                ['none', 'wooden_pickaxe'], _other='none', _default='none'), ]

    # john rl nyu microsfot van roy and ian osband

    def create_rewardables(self) -> List[Handler]:
        reward_handler = (
            handlers.RewardForCollectingItems if self.dense
            else handlers.RewardForCollectingItemsOnce)

        return [
            reward_handler(self.reward_schedule if self.reward_schedule else {
                           self.target_item: 1})
        ]

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type='log', quantity='8')
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type='wooden_pickaxe', amount=1)
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
            handlers.ServerQuitFromTimeUp(CRAFT_STEPS * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

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
        # TODO: Convert this to finish handlers.
        rewards = set(rewards)
        allow_missing_ratio = 0.1
        max_missing = round(len(self.reward_schedule) * allow_missing_ratio)

        # Get a list of the rewards from the reward_schedule.
        reward_values = [
            s['reward'] for s in self.reward_schedule
        ]

        return len(rewards.intersection(reward_values)) \
            >= len(reward_values) - max_missing
