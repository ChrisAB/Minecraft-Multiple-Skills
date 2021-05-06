from minerl.herobraine.env_specs.obtain_specs import Obtain


class ObtainStick(Obtain):

    def __init__(self, dense=False, *args, **kwargs):
        super(ObtainStick, self).__init__(*args,
                                          target_item='stick',
                                          dense=dense,
                                          reward_schedule=[
                                              dict(type="stick",
                                                   amount=1, reward=4)
                                          ],
                                          max_episode_steps=18000,
                                          **kwargs
                                          )

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'o_stick'

    def get_docstring(self):
        return ""
