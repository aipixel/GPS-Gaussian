import numpy as np
import taichi as ti
import taichi.math as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class ScatterModel(AutoInit):
    def __init__(self, num=None, radius=2):
        self.L2W = Affine.field(())

        self.num = num
        self.radius = radius

        if num is not None:
            self.particles = ti.Vector.field(3, ti.i32, num)

    def _init(self):
        self.L2W.init()

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.particles):
            render_particle(self, camera, self.particles[i], self.radius)
