import numpy as np
import taichi as ti
import taichi_glsl as tl
import math


def _pre(x):
    if not isinstance(x, ti.Matrix):
        x = ti.Vector(x)
    return x


def _ser(foo):
    def wrapped(self, *args, **kwargs):
        foo(self, *args, **kwargs)
        return self

    return wrapped


def _mparg(foo):
    def wrapped(self, *args):
        if len(args) > 1:
            return [foo(self, x) for x in args]
        else:
            return foo(self, args[0])

    return wrapped


class MeshGen:
    def __init__(self):
        self.v = []
        self.f = []

    @_ser
    def quad(self, a, b, c, d):
        a, b, c, d = self.add_v(a, b, c, d)
        self.add_f([a, b, c], [c, d, a])

    @_ser
    def cube(self, a, b):
        aaa = self.add_v(tl.mix(a, b, tl.D.yyy))
        baa = self.add_v(tl.mix(a, b, tl.D.xyy))
        aba = self.add_v(tl.mix(a, b, tl.D.yxy))
        aab = self.add_v(tl.mix(a, b, tl.D.yyx))
        bba = self.add_v(tl.mix(a, b, tl.D.xxy))
        abb = self.add_v(tl.mix(a, b, tl.D.yxx))
        bab = self.add_v(tl.mix(a, b, tl.D.xyx))
        bbb = self.add_v(tl.mix(a, b, tl.D.xxx))

        self.add_f4([aaa, aba, bba, baa]) # back
        self.add_f4([aab, bab, bbb, abb]) # front
        self.add_f4([aaa, aab, abb, aba]) # left
        self.add_f4([baa, bba, bbb, bab]) # right
        self.add_f4([aaa, baa, bab, aab]) # bottom
        self.add_f4([aba, abb, bbb, bba]) # top

    @_ser
    def cylinder(self, bottom, top, dir1, dir2, N):
        bottom = _pre(bottom)
        top = _pre(top)
        dir1 = _pre(dir1)
        dir2 = _pre(dir2)

        B, T = [], []
        for i in range(N):
            disp = tl.mat(dir1.entries, dir2.entries).T() @ tl.vecAngle(tl.math.tau * i / N)
            B.append(self.add_v(bottom + disp))
            T.append(self.add_v(top + disp))

        BC = self.add_v(bottom)
        TC = self.add_v(top)

        for i in range(N):
            j = (i + 1) % N
            self.add_f4([B[i], B[j], T[j], T[i]])

        for i in range(N):
            j = (i + 1) % N
            self.add_f([B[j], B[i], BC])
            self.add_f([T[i], T[j], TC])

    @_ser
    def tri(self, a, b, c):
        a, b, c = self.add_v(a, b, c)
        self.add_f([a, b, c])

    @_mparg
    def add_v(self, v):
        if isinstance(v, ti.Matrix):
            v = v.entries
        ret = len(self.v)
        self.v.append(v)
        return ret

    @_mparg
    def add_f(self, f):
        ret = len(self.f)
        self.f.append(f)
        return ret

    @_mparg
    def add_f4(self, f):
        a, b, c, d = f
        return self.add_f([a, b, c], [c, d, a])


    def __getitem__(self, key):
        if key == 'v':
            return np.array(self.v)
        if key == 'f':
            return np.array(self.f)
