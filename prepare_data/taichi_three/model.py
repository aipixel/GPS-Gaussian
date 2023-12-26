import numpy as np
import taichi as ti
import taichi.math as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.func
def sample(field: ti.template(), P):
    '''
    Sampling a field with indices clampped into the field shape.
    :parameter field: (Tensor)
        Specify the field to sample.
    :parameter P: (Vector)
        Specify the index in field.
    :return:
        The return value is calcuated as::
            P = clamp(P, 0, vec(*field.shape) - 1)
            return field[int(P)]
    '''
    shape = ti.Vector(field.shape)
    P = ts.clamp(P, 0, shape - 1)
    return field[int(P)]

@ti.func
def bilerp(field: ti.template(), P):
    '''
    Bilinear sampling an 2D field with a real index.
    :parameter field: (2D Tensor)
        Specify the field to sample.
    :parameter P: (2D Vector of float)
        Specify the index in field.
    :note:
        If one of the element to be accessed is out of `field.shape`, then
        `bilerp` will automatically do a clamp for you, see :func:`sample`. 
    :return:
        The return value is calcuated as::
            I = int(P)
            x = fract(P)
            y = 1 - x
            return (sample(field, I + D.xx) * x.x * x.y +
                    sample(field, I + D.xy) * x.x * y.y +
                    sample(field, I + D.yy) * y.x * y.y +
                    sample(field, I + D.yx) * y.x * x.y)
        .. where D = vec(1, 0, -1)
    '''
    I = int(P)
    x = ts.fract(P)
    y = 1 - x
    D = ts.vec3(1, 0, -1)
    return (sample(field, I + D.xx) * x.x * x.y +
            sample(field, I + D.xy) * x.x * y.y +
            sample(field, I + D.yy) * y.x * y.y +
            sample(field, I + D.yx) * y.x * x.y)

@ti.data_oriented
class Model(AutoInit):
    TEX = 0
    COLOR = 1

    def __init__(self, f_n=None, f_m=None,
            vi_n=None, vt_n=None, vn_n=None, tex_n=None, col_n=None,
            obj=None, tex=None):

        self.faces = None
        self.vi = None
        self.vt = None
        self.vn = None
        self.tex = None
        self.type = ti.field(dtype=ti.int32, shape=())
        self.reverse = False

        if obj is not None:
            f_n = None if obj['f'] is None else obj['f'].shape[0]
            vi_n = None if obj['vi'] is None else obj['vi'].shape[0]
            vt_n = None if obj['vt'] is None else obj['vt'].shape[0]
            vn_n = None if obj['vn'] is None else obj['vn'].shape[0]

        if tex is not None:
            tex_n = tex.shape[:2]

        if f_m is None:
            f_m = 1
            if vt_n is not None:
                f_m = 2
            if vn_n is not None:
                f_m = 3

        if vi_n is None:
            vi_n = 1
        if vt_n is None:
            vt_n = 1
        if vn_n is None:
            vn_n = 1
        if col_n is None:
            col_n = 1

        if f_n is not None:
            self.faces = ti.Matrix.field(3, f_m, ti.i32, f_n)
        if vi_n is not None:
            self.vi = ti.Vector.field(3, ti.f32, vi_n)
        if vt_n is not None:
            self.vt = ti.Vector.field(2, ti.f32, vt_n)
        if vn_n is not None:
            self.vn = ti.Vector.field(3, ti.f32, vn_n)
        if tex_n is not None:
            self.tex = ti.Vector.field(3, ti.f32, tex_n)
        if col_n is not None:
            self.vc = ti.Vector.field(3, ti.f32, col_n)

        if obj is not None:
            self.init_obj = obj
        if tex is not None:
            self.init_tex = tex

    def from_obj(self, obj):
        if obj['f'] is not None:
            self.faces.from_numpy(obj['f'])
        if obj['vi'] is not None:
            self.vi.from_numpy(obj['vi'])
        if obj['vt'] is not None:
            self.vt.from_numpy(obj['vt'])
        if obj['vn'] is not None:
            self.vn.from_numpy(obj['vn'])

    def _init(self):
        self.type[None] = 0
        if hasattr(self, 'init_obj'):
            self.from_obj(self.init_obj)
        if hasattr(self, 'init_tex'):
            self.tex.from_numpy(self.init_tex.astype(np.float32) / 255)

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def texSample(self, coor):
        if ti.static(self.tex is not None):
            return ts.bilerp(self.tex, coor * ts.vec(*self.tex.shape))
        else:
            return 1


@ti.data_oriented
class StaticModel(AutoInit):
    TEX = 0
    COLOR = 1

    def __init__(self, N, f_m=None, col_n=None,
            obj=None, tex=None):
        self.faces = None
        self.vi = None
        self.vt = None
        self.vn = None
        self.tex = None
        # 0 origin 1 pure color 2 shader color
        self.type = ti.field(dtype=ti.int32, shape=())
        self.f_n = ti.field(dtype=ti.int32, shape=())
        self.reverse = False
        self.N = N

        if obj is not None:
            f_n = None if obj['f'] is None else obj['f'].shape[0]
            vi_n = None if obj['vi'] is None else obj['vi'].shape[0]
            vt_n = None if obj['vt'] is None else obj['vt'].shape[0]
            vn_n = None if obj['vn'] is None else obj['vn'].shape[0]

        if not (tex is None):
            tex_n = tex.shape[:2]
        else:
            tex_n = None

        if f_m is None:
            f_m = 1
            if vt_n is not None:
                f_m = 2
            if vn_n is not None:
                f_m = 3

        if vi_n is None:
            vi_n = 1
        if vt_n is None:
            vt_n = 1
        if vn_n is None:
            vn_n = 1
        if col_n is None:
            col_n = 1

        if f_n is not None:
            self.faces = ti.Matrix.field(3, f_m, ti.i32, N)
        if vi_n is not None:
            self.vi = ti.Vector.field(3, ti.f32, N)
        if vt_n is not None:
            self.vt = ti.Vector.field(2, ti.f32, N)
        if vn_n is not None:
            self.vn = ti.Vector.field(3, ti.f32, N)
        if not (tex_n is None):
            self.tex = ti.Vector.field(3, ti.f32, tex_n)
        if col_n is not None:
            self.vc = ti.Vector.field(3, ti.f32, N)

        if obj is not None:
            self.init_obj = obj
        if tex is not None:
            self.init_tex = tex

    def modify_color(self, color):
        s_color = np.zeros((self.N, 3)).astype(np.float32)
        s_color[:color.shape[0]] = color
        self.vc.from_numpy(s_color)

    def from_obj(self, obj):
        N = self.N
        if obj['f'] is not None:
            s_faces = np.zeros((N, obj['f'].shape[1], obj['f'].shape[2])).astype(int)
            s_faces[:obj['f'].shape[0]] = obj['f']
            self.f_n[None] = obj['f'].shape[0]
            self.faces.from_numpy(s_faces)
        if obj['vi'] is not None:
            s_vi = np.zeros((N, 3)).astype(np.float32)
            s_vi[:obj['vi'].shape[0]] = obj['vi'][:, :3]
            self.vi.from_numpy(s_vi)
        if obj['vt'] is not None:
            s_vt = np.zeros((N, 2)).astype(np.float32)
            s_vt[:obj['vt'].shape[0]] = obj['vt']
            self.vt.from_numpy(s_vt)
        if obj['vn'] is not None:
            s_vn = np.zeros((N, 3)).astype(np.float32)
            s_vn[:obj['vn'].shape[0]] = obj['vn']
            self.vn.from_numpy(s_vn)

    def _init(self):
        self.type[None] = 0
        if hasattr(self, 'init_obj'):
            self.from_obj(self.init_obj)
        if hasattr(self, 'init_tex') and (self.init_tex is not None):
            self.tex.from_numpy(self.init_tex.astype(np.float32) / 255)

    @ti.func
    def render(self, camera, lights):
        for i in ti.grouped(self.faces):
            render_triangle(self, camera, self.faces[i], lights)

    @ti.func
    def texSample(self, coor):
        if ti.static(self.tex is not None):
            return bilerp(self.tex, coor * ts.vec2(*self.tex.shape))
        else:
            return 1
