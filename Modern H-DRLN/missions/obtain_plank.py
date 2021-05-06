from minerl.herobraine.env_specs.obtain_specs import Obtain


class ObtainPlank(Obtain):

    def __init__(self, dense=False, *args, **kwargs):
        super(ObtainPlank, self).__init__(*args,
                                          target_item='plank',
                                          dense=dense,
                                          reward_schedule=[
                                              dict(type="plank",
                                                   amount=1, reward=4)
                                          ],
                                          max_episode_steps=18000,
                                          **kwargs
                                          )

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'o_plank'

    def get_docstring(self):
        return ""
