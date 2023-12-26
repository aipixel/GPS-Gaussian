import math
import taichi as ti
import taichi.math as ts


@ti.func
def render_triangle(model, camera, face, lights):
    scene = model.scene
    _1 = ti.static(min(1, model.faces.m - 1))
    _2 = ti.static(min(2, model.faces.m - 1))
    ia, ib, ic = model.vi[face[0, 0]], model.vi[face[1, 0]], model.vi[face[2, 0]]
    ca, cb, cc = ia, ib, ic
    if model.type[None] >= 1:
        ca, cb, cc = model.vc[face[0, 0]], model.vc[face[1, 0]], model.vc[face[2, 0]]
    ta, tb, tc = model.vt[face[0, _1]], model.vt[face[1, _1]], model.vt[face[2, _1]]
    na, nb, nc = model.vn[face[0, _2]], model.vn[face[1, _2]], model.vn[face[2, _2]]
    a = camera.untrans_pos(ia)
    b = camera.untrans_pos(ib)
    c = camera.untrans_pos(ic)

    # NOTE: the normal computation indicates that # a front-facing face should
    # be COUNTER-CLOCKWISE, i.e., glFrontFace(GL_CCW);
    # this is to be compatible with obj model loading.
    normal = (a - b).cross(a - c).normalized()
    pos = (a + b + c) / 3
    view_pos = (a + b + c) / 3
    shading = (view_pos.normalized().dot(normal) - 0.5)*2.0
    if ti.static(camera.type == camera.ORTHO):
        view_pos = ti.Vector([0.0, 0.0, 1.0], ti.f32)
    reverse_v = 1
    if ti.static(model.reverse == True):
        reverse_v = -1
    if ts.dot(view_pos, normal)*reverse_v <= 0:
        # shading
        color = ti.Vector([0.0, 0.0, 0.0], ti.f32)
        for i in range(lights.n):
            light_dir = lights.light_dirs[i]
            # print(light_dir)
            light_dir = camera.untrans_dir(light_dir)
            light_color = lights.light_colors[i]
            color += scene.opt.render_func(pos, normal, ti.Vector([0.0, 0.0, 0.0], ti.f32), light_dir, light_color)
        color = scene.opt.pre_process(color)
        A = camera.uncook(a)
        B = camera.uncook(b)
        C = camera.uncook(c)
        scr_norm = 1 / (A - C).cross(B - A)
        B_A = (B - A) * scr_norm
        C_B = (C - B) * scr_norm
        A_C = (A - C) * scr_norm

        W = 1
        # screen space bounding box
        M, N = int(ti.floor(min(A, B, C) - W)), int(ti.ceil(max(A, B, C) + W))
        M.x, N.x = min(max(M.x, 0), camera.img.shape[0]), min(max(N.x, 0), camera.img.shape[1])
        M.y, N.y = min(max(M.y, 0), camera.img.shape[0]), min(max(N.y, 0), camera.img.shape[1])
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            # barycentric coordinates using the area method
            X_A = X - A
            w_C = B_A.cross(X_A)
            w_B = A_C.cross(X_A)
            w_A = 1 - w_C - w_B
            # draw
            in_screen = w_A >= 0 and w_B >= 0 and w_C >= 0 and 0 < X[0] < camera.img.shape[0] and 0 < X[1] < camera.img.shape[1]
            if not in_screen:
                continue
            zindex = 0.0
            if ti.static(model.reverse == True):
                zindex = (a.z * w_A + b.z * w_B + c.z * w_C)
            else:
                zindex = 1 / (a.z * w_A + b.z * w_B + c.z * w_C)
            if zindex < ti.atomic_max(camera.zbuf[X], zindex):
                continue
            
            coor = (ta * w_A + tb * w_B + tc * w_C)
            if model.type[None] == 1:
                camera.img[X] = (ca * w_A + cb * w_B + cc * w_C)
            elif model.type[None] == 2:
                camera.img[X] = color * (ca * w_A + cb * w_B + cc * w_C)
            else:    
                camera.img[X] = color * model.texSample(coor)
            camera.mask[X] = ti.Vector([1.0, 1.0, 1.0], ti.f32)
            camera.normal_map[X] = -normal / 2.0 + 0.5
            # camera.normal_map[X] = ts.vec3(shading / 2.0 + 0.5)

@ti.func
def render_particle(model, camera, vertex, radius):
    scene = model.scene
    L2W = model.L2W
    a = camera.untrans_pos(L2W @ vertex)
    A = camera.uncook(a)

    M = int(ti.floor(A - radius))
    N = int(ti.ceil(A + radius))

    for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
        if X.x < 0 or X.x >= camera.res[0] or X.y < 0 or X.y >= camera.res[1]:
            continue
        if (X - A).norm_sqr() > radius**2:
            continue

        camera.img[X] = ti.Vector([1.0, 1.0, 1.0], ti.f32)
