import torch
from torch import nn
from torch.nn import functional as F

import extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, n_classes)

        print(psp_size, int(psp_size / len(sizes)))
        self.psp = PSPModule(psp_size, int(psp_size / len(sizes)), sizes)

        self.ppm = PPM(psp_size, int(psp_size / len(sizes)), sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.drop_2 = nn.Dropout2d(p=0.15)

        self.final = nn.Sequential(
            nn.Conv2d(psp_size*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, n_classes, kernel_size=1)
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

        p = self.ppm(f)
        print('After Psp')
        print(p.shape)
        print(class_f.shape)

        p = self.final(p)

        p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=True)
        # aux branch
        aux = self.aux(class_f)

        aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

        return p, aux

if __name__ == '__main__':
    input = torch.rand(4, 3, 64, 64)
    model = PSPNet(n_classes=19, pretrained=False, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50')
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output[0].size())
    print('PSPNet', output[1].size())
