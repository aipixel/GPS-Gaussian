import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class UnetExtractor(nn.Module):
    def __init__(self, in_channel=3, encoder_dim=[64, 96, 128], norm_fn='group'):
        super().__init__()
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            ResidualBlock(32, encoder_dim[0], norm_fn=norm_fn),
            ResidualBlock(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn)
        )
        self.res2 = nn.Sequential(
            ResidualBlock(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn)
        )
        self.res3 = nn.Sequential(
            ResidualBlock(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )

    def forward(self, x):
        x = self.in_ds(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x1, x2, x3


class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], encoder_dim=[64, 96, 128]):
        super(MultiBasicEncoder, self).__init__()

        # output convolution for feature
        self.conv2 = nn.Sequential(
            ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
            nn.Conv2d(encoder_dim[2], encoder_dim[2]*2, 3, padding=1))

        # output convolution for context
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
                nn.Conv2d(encoder_dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

    def forward(self, x):
        feat1, feat2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)

        outputs08 = [f(x) for f in self.outputs08]
        return outputs08, feat1, feat2


if __name__ == '__main__':

    data = torch.ones((1, 3, 1024, 1024))

    model = UnetExtractor(in_channel=3, encoder_dim=[64, 96, 128])

    x1, x2, x3 = model(data)
    print(x1.shape, x2.shape, x3.shape)
