import taichi as ti
import taichi.math as tm
from .transform import *
import math

class Shading:
    def __init__(self, **kwargs):
        self.is_normal_map = False
        self.lambert = 0.58
        self.half_lambert = 0.04
        self.blinn_phong = 0.3
        self.phong = 0.0
        self.shineness = 10
        self.__dict__.update(kwargs)

    @ti.func
    def render_func(self, pos, normal, dir, light_dir, light_color):
        color = ti.Vector([0.0, 0.0, 0.0], ti.f32)
        shineness = self.shineness
        half_lambert = normal.dot(light_dir) * 0.5 + 0.5
        lambert = max(0, normal.dot(light_dir))
        blinn_phong = normal.dot(tm.mix(light_dir, -dir, 0.5))
        blinn_phong = pow(max(blinn_phong, 0), shineness)
        refl_dir = tm.reflect(light_dir, normal)
        phong = -tm.dot(normal, refl_dir)
        phong = pow(max(phong, 0), shineness)

        strength = 0.0
        if ti.static(self.lambert != 0.0):
            strength += lambert * self.lambert
        if ti.static(self.half_lambert != 0.0):
            strength += half_lambert * self.half_lambert
        if ti.static(self.blinn_phong != 0.0):
            strength += blinn_phong * self.blinn_phong
        if ti.static(self.phong != 0.0):
            strength += phong * self.phong
        color = tm.vec3(strength)

        if ti.static(self.is_normal_map):
            color = normal * 0.5 + 0.5
        return color * light_color

    @ti.func
    def pre_process(self, color):
        blue = tm.vec3(0.00, 0.01, 0.05)
        orange = tm.vec3(1.19, 1.04, 0.98)
        return ti.sqrt(ts.mix(blue, orange, color))
