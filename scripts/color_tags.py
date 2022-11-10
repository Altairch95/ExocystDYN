from pymol.cgo import *
from pymol import cmd


@cmd.extend
def color_tags(sphere_size=1):
    cmd.hide("all")
    cmd.show_as("spheres")
    cmd.set("sphere_scale", value=int(sphere_size))  # value referred to sphere size
    cmd.color("marine", "chain A")
    cmd.color("orange", "chain B")
    cmd.color("yellow", "chain C")
    cmd.color("pink", "chain D")
    cmd.color("chocolate", "chain E")
    cmd.color("purple", "chain F")
    cmd.color("red", "chain G")
    cmd.color("green", "chain H")
    cmd.color("grey70", "chain I")
    cmd.bg_color('black')

