from .find_tree import NavigateTree
from .obtain_log import ObtainLog
from .craft_wooden_pickaxe import CraftWoodenPickaxe
from .obtain_stick import ObtainStick

NavigateTree().register()
ObtainLog().register()
CraftWoodenPickaxe().register()
ObtainStick().register()
