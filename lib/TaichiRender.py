
import taichi as ti
from lib.utils import *
ti.init(ti.cuda)


@ti.data_oriented
class TaichiRenderBatch:
    def __init__(self, bs, res):
        self.res = res
        self.coord = ti.Vector.field(n=1, dtype=ti.i32, shape=(bs, res * res))

    @ti.kernel
    def render_respective_color(self, pts: ti.types.ndarray(), pts_mask: ti.types.ndarray(),
                                render_depth: ti.types.ndarray(), render_color: ti.types.ndarray()):
        for B, i in self.coord:
            if pts_mask[B, i, 0] < 0.5:
                continue
            IX, IY = ti.cast(pts[B, i, 0], ti.i32), ti.cast(pts[B, i, 1], ti.i32)
            IX = ti.min(self.res - 1, ti.max(IX, 0))
            IY = ti.min(self.res - 1, ti.max(IY, 0))
            if pts[B, i, 2] >= ti.atomic_max(render_depth[B, 0, IY, IX], pts[B, i, 2]):
                for k in ti.static(range(3)):
                    render_color[B, k, IY, IX] = pts[B, i, k + 3]

    def flow2render(self, data):
        novel_view_calib = torch.matmul(data['novel_view']['intr'], data['novel_view']['extr'])
        B = novel_view_calib.shape[0]

        taichi_pts_list = []
        taichi_mask_list = []

        for view in ['lmain', 'rmain']:
            data_select = data[view]
            depth_pred = flow2depth(data_select).clone()
            valid = depth_pred != 0

            pts = depth2pc(depth_pred, data_select['extr'], data_select['intr'])
            valid = valid.view(B, -1, 1).squeeze(2)
            pts_valid = torch.zeros_like(pts)
            pts_valid[valid] = pts[valid]

            pts_valid = perspective(pts_valid, novel_view_calib)
            pts_valid[:, :, 2:] = 1.0 / (pts_valid[:, :, 2:] + 1e-8)

            img_valid = torch.zeros_like(pts_valid)
            img_valid[valid] = data_select['img'].permute(0, 2, 3, 1).view(B, -1, 3)[valid]
            taichi_pts = torch.cat((pts_valid, img_valid), dim=2)
            taichi_mask = valid.view(B, -1, 1).float()
            taichi_pts_list.append(taichi_pts)
            taichi_mask_list.append(taichi_mask)

        render_depth = torch.zeros((B, 1, self.res, self.res), device=pts.device).float()
        min_value = -1
        render_color = min_value + torch.zeros((B, 3, self.res, self.res), device=pts.device).float()
        for i in range(2):
            self.render_respective_color(taichi_pts_list[i], taichi_mask_list[i], render_depth, render_color)
        data['novel_view']['img_pred'] = render_color

        return data
