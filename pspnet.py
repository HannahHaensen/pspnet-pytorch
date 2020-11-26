import torch
from torch import nn
from torch.nn import functional as F

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(size)
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)

        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        return torch.cat(priors, 1)
        # bottle = self.bottleneck(torch.cat(priors, 1))

        # return self.relu(bottle)
        # out = [feats]


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50', pretrained=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, n_classes)

        self.psp = PSPModule(psp_size, int(psp_size / len(sizes)), sizes)

        self.final = nn.Sequential(
            nn.Conv2d(psp_size*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = x_size[2]
        w = x_size[3]

        f, class_f = self.feats(x)

        p = self.psp(f)
        print('After Psp')
        print(p.shape)

        p = self.final(p)

        p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=True)
        # aux branch
        aux = self.aux(class_f)

        aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

        return p, aux

if __name__ == '__main__':
    input = torch.rand(4, 3, 64, 128)
    model = PSPNet(n_classes=19, pretrained=False, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50')
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output[0].size())
    print('PSPNet', output[1].size())
