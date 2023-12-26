import taichi as ti
from .transform import *
from .shading import *
from .light import *


@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.lights = []
        self.cameras = []
        self.opt = Shading()
        self.models = []

    @ti.func
    def cook_coor(self, I, camera):
        scale = ti.static(2 / min(*camera.img.shape()))
        coor = (I - ts.vec2(*camera.img.shape()) / 2) * scale
        return coor

    @ti.func
    def uncook_coor(self, coor, camera):
        scale = ti.static(min(*camera.img.shape()) / 2)
        I = coor.xy * scale + ts.vec2(*camera.img.shape()) / 2
        return I

    def add_model(self, model):
        model.scene = self
        self.models.append(model)

    def add_camera(self, camera):
        camera.scene = self
        self.cameras.append(camera)

    def add_lights(self, lights):
        lights.scene = self
        self.lights = lights

    def _init(self):
        for camera in self.cameras:
            camera.init()
        for model in self.models:
            model.init()
        self.lights.init_data()

    
    @ti.kernel
    def _single_render(self, num : ti.template()):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                camera.clear_buffer()
                self.models[num].render(camera, self.lights)

    def single_render(self, num):
        self.lights.init_data()
        for camera in self.cameras:
            camera.init()
        self.models[num].init()
        self._single_render(num)

    def render(self):
        self.init()
        self._render()

    @ti.kernel
    def _render(self):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                camera.clear_buffer()
                # sets up light directions
                if ti.static(len(self.models)):
                    for model in ti.static(self.models):
                        model.render(camera, self.lights)
