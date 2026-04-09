import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import densenet121, DenseNet121_Weights

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, dim_in),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.1),
                                    nn.Linear(dim_in, out_dim))

    def forward(self, x):
        return self.layers(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim, bias=False))

    def forward(self, x):
        z = self.proj(x)
        return z

class MyDenseNet(nn.Module):
    def __init__(self, num_classes, useBothEyes, proj_dim):
        super().__init__()
        self.features = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        nfea = self.features.classifier.in_features
        self.features.classifier = Identity()

        self.useBothEyes = useBothEyes
        times = 2 if useBothEyes else 1
        feature_dim = nfea * times

        self.dout = nn.Dropout(p=0.2)

        self.mse_feat = MLP(feature_dim, feature_dim)
        self.ce_feat = MLP(feature_dim, feature_dim)

        self.fc_mse = nn.Linear(feature_dim, 1)
        self.fc_ce = nn.Linear(feature_dim, num_classes)
        self.projection_head = ProjectionHead(feature_dim, proj_dim)


    def forward(self, x1, x2=None):
        if self.useBothEyes:
            x1 = self.features(x1)
            x2 = self.features(x2)
            x = torch.cat((x1, (x1 + x2) / 2), dim=1)
        else:
            x = self.features(x1)

        x = self.dout(x)
        z = F.normalize(self.projection_head(x), dim=1)

        mse_fea = self.mse_feat(x)
        ce_fea = self.ce_feat(x)

        mse_out = self.fc_mse(mse_fea)
        ce_out = self.fc_ce(ce_fea)

        return z, mse_out, ce_out























