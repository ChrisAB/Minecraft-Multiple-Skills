from .find_tree import NavigateTree
from .obtain_log import ObtainLog
from .obtain_plank import ObtainPlank
from .obtain_stick import ObtainStick

NavigateTree().register()
ObtainLog().register()
ObtainPlank().register()
ObtainStick().register()
