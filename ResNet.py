import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim_in, out_dim)
        )
    def forward(self, x):
        return self.layers(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim, bias=False),
        )
    def forward(self, x):
        return self.proj(x)


class MyResNet(nn.Module):
    def __init__(self, num_classes, useBothEyes, proj_dim):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        nfea = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.useBothEyes = useBothEyes
        times =  2 if useBothEyes else 1
        feature_dim = nfea * times

        self.dout = nn.Dropout(p=0.2)

        self.mse_feat = MLP(feature_dim, feature_dim)
        self.ce_feat = MLP(feature_dim, feature_dim)

        self.fc_mse = nn.Linear(feature_dim, 1)
        self.fc_ce = nn.Linear(feature_dim, num_classes)

        self.projection_head = ProjectionHead(feature_dim, proj_dim)

    def extract(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2=None):
        x1 = self.extract(x1)

        if self.useBothEyes:
            x2 = self.extract(x2)
            raw_feats = torch.cat((x1, (x1 + x2) / 2), dim=1)
        else:
            raw_feats = x1

        z = self.projection_head(raw_feats)
        z = F.normalize(z, dim=1)

        mse_fea = self.mse_feat(raw_feats)
        ce_fea = self.ce_feat(raw_feats)

        mse_out = self.fc_mse(mse_fea)
        ce_out = self.fc_ce(ce_fea)

        return z, mse_out, ce_out