
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def get_novel_calib(data, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


def get_novel_calib_for_show(data, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    intr_list, extr_list = [], []
    data['novel_view'] = {}
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]

        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)
        intr_list.append(intr_new)
        extr_list.append(extr_new)
    data['novel_view']['intr'] = torch.FloatTensor(np.array(intr_list)).cuda()
    data['novel_view']['extr'] = torch.FloatTensor(np.array(extr_list)).cuda()
    return data


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def flow2depth(data):
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    offset = torch.broadcast_to(offset[:, None, None, None], data['flow_pred'].shape)
    disparity = offset - data['flow_pred']
    depth = -disparity / data['Tf_x'][:, None, None, None]
    depth *= data['mask'][:, :1, :, :]

    return depth

def perspective(pts, calibs):
    pts = pts.permute(0, 2, 1)
    pts = torch.bmm(calibs[:, :3, :3], pts)
    pts = pts + calibs[:, :3, 3:4]
    pts[:, :2, :] /= pts[:, 2:, :]
    pts = pts.permute(0, 2, 1)
    return pts
