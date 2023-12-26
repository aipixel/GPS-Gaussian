import taichi as ti
import taichi.math as ts
from .scene import *
import math


EPS = 1e-3
INF = 1e3


@ti.data_oriented
class ObjectRT(ts.TaichiClass):
    @ti.func
    def calc_sdf(self, p):
        ret = INF
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            ret = min(ret, self.make_one(I).do_calc_sdf(p))
        return ret

    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            t, n = self.make_one(I).do_intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal

    def do_calc_sdf(self, p):
        raise NotImplementedError

    def do_intersect(self, orig, dir):
        raise NotImplementedError


@ti.data_oriented
class Ball(ObjectRT):
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    @ti.func
    def make_one(self, I):
        return Ball(self.pos[I], self.radius[I])

    @ti.func
    def do_calc_sdf(self, p):
        return ts.distance(self.pos, p) - self.radius

    @ti.func
    def do_intersect(self, orig, dir):
        op = self.pos - orig
        b = op.dot(dir)
        det = b ** 2 - op.norm_sqr() + self.radius ** 2
        ret = INF
        if det > 0.0:
            det = ti.sqrt(det)
            t = b - det
            if t > EPS:
                ret = t
            else:
                t = b + det
                if t > EPS:
                    ret = t
        return ret, ts.normalize(dir * ret - op)


@ti.data_oriented
class SceneRTBase(Scene):
    def __init__(self):
        super(SceneRTBase, self).__init__()
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    @ti.func
    def color_at(self, coor, camera):
        orig, dir = camera.generate(coor)

        pos, normal = self.trace(orig, dir)
        light_dir = self.light_dir[None]

        color = self.opt.render_func(pos, normal, dir, light_dir)
        color = self.opt.pre_process(color)
        return color

    @ti.kernel
    def _render(self):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                for I in ti.grouped(camera.img):
                    coor = self.cook_coor(I, camera)
                    color = self.color_at(coor, camera)
                    camera.img[I] = color

    def add_ball(self, pos, radius):
        b = Ball(pos, radius)
        self.balls.append(b)


@ti.data_oriented
class SceneRT(SceneRTBase):
    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for b in ti.static(self.balls):
            t, n = b.intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal

    @ti.func
    def trace(self, orig, dir):
        depth, normal = self.intersect(orig, dir)
        pos = orig + dir * depth
        return pos, normal


@ti.data_oriented
class SceneSDF(SceneRTBase):
    @ti.func
    def calc_sdf(self, p):
        ret = INF
        for b in ti.static(self.balls):
            ret = min(ret, b.calc_sdf(p))
        return ret

    @ti.func
    def calc_grad(self, p):
        return ts.vec(
            self.calc_sdf(p + ts.vec(EPS, 0, 0)),
            self.calc_sdf(p + ts.vec(0, EPS, 0)),
            self.calc_sdf(p + ts.vec(0, 0, EPS)))

    @ti.func
    def trace(self, orig, dir):
        pos = orig
        color = ts.vec3(0.0)
        normal = ts.vec3(0.0)
        for s in range(100):
            t = self.calc_sdf(pos)
            if t <= 0:
                normal = ts.normalize(self.calc_grad(pos) - t)
                break
            pos += dir * t
        return pos, normal
