from minerl.herobraine.env_specs.obtain_specs import Obtain


class ObtainLog(Obtain):

    def __init__(self, dense=False, *args, **kwargs):
        super(ObtainLog, self).__init__(*args,
                                        target_item='log',
                                        dense=dense,
                                        reward_schedule=[
                                            dict(type="log",
                                                 amount=1, reward=4)
                                        ],
                                        max_episode_steps=18000,
                                        **kwargs
                                        )

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'o_log'

    def get_docstring(self):
        return ""
