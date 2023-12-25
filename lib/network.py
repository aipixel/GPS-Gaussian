
import torch
from torch import nn
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        self.raft_stereo = RAFTStereoHuman(self.cfg.raft)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        flow = torch.cat([data['lmain']['flow'], data['rmain']['flow']], dim=0) if is_train else None
        valid = torch.cat([data['lmain']['valid'], data['rmain']['valid']], dim=0) if is_train else None

        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)

        if is_train:
            flow_predictions = self.raft_stereo(img_feat[2], iters=self.train_iters)
            flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

        else:
            flow_up = self.raft_stereo(img_feat[2], iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['depth'] != 0.0
            data[view]['pts_valid'] = valid.view(bs, -1)

        # regress gaussian parms
        lr_depth = torch.concat([data['lmain']['depth'], data['rmain']['depth']], dim=0)
        rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data

