import taichi as ti
from .common import *
import math

'''
The base light class represents a directional light.
'''
@ti.data_oriented
class Light(AutoInit):

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [x / norm for x in dir]
 
        self.dir_py = [-x for x in dir]
        self.color_py = color or [1, 1, 1] 

        self.dir = ti.Vector.field(3, ti.float32, ())
        self.color = ti.Vector.field(3, ti.float32, ())
        # store the current light direction in the view space
        # so that we don't have to compute it for each vertex
        self.viewdir = ti.Vector.field(3, ti.float32, ())

    def set(self, dir=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x**2 for x in dir))
        dir = [x / norm for x in dir]
        self.dir_py = dir
        self.color = color

    def _init(self):
        self.dir[None] = self.dir_py
        self.color[None] = self.color_py

    @ti.func
    def intensity(self, pos):
        return 1

    @ti.func
    def get_color(self):
        return self.color[None]

    @ti.func
    def get_dir(self):
        return self.viewdir

    @ti.func
    def set_view(self, camera):
        self.viewdir[None] = camera.untrans_dir(self.dir[None])


@ti.data_oriented
class Lights(AutoInit):

    def __init__(self, light_list):
        n = len(light_list)
        self.n = n
        self.light_list = light_list
        self.light_dirs = ti.Vector.field(3, ti.float32, n)
        self.light_colors = ti.Vector.field(3, ti.float32, n)

    def init_data(self):
        index = 0
        for light in self.light_list:
            self.light_dirs[index] = self.light_list[index].dir_py
            self.light_colors[index] = self.light_list[index].color_py
            index += 1

class PointLight(Light):
    pass

