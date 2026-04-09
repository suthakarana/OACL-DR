import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.manifold import TSNE
from Models.DataLoader import *


class TSNEPlot(nn.Module):
    def __init__(self, opt, model_path, split='test'):
        super(TSNEPlot, self).__init__()
        self.opt = opt
        self.model_path = model_path
        self.split = split

        self.trainLoader, self.testLoader, self.valLoader, self.uniqueLbls = getDataLoaders(
            opt.dataset, opt.pL, opt.seed, opt.bs_Tr, opt.bs_Te, opt.bs_Val
        )

        self.nClass = len(self.uniqueLbls)
        self.model = self.load_model()

    def load_model(self):
        ckpt = torch.load(self.model_path)

        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = ckpt['model']
        else:
            model = ckpt

        model = model.cuda(self.opt.gpuid)
        model.eval()
        return model

    def get_loader(self):
        if self.split == 'train':
            return self.trainLoader
        elif self.split == 'val':
            return self.valLoader
        elif self.split == 'test':
            return self.testLoader
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

    @torch.no_grad()
    def extract_features(self):
        loader = self.get_loader()

        features = []
        labels = []

        for i, (I1, I2, y, _) in enumerate(loader):
            I1 = Variable(I1.cuda(self.opt.gpuid))
            I2 = Variable(I2.cuda(self.opt.gpuid))
            y = y.cuda(self.opt.gpuid)

            z, _, _ = self.model(I1, I2)
            z = F.normalize(z, dim=1)

            features.append(z.cpu())
            labels.append(y.cpu())

            print(f"\rExtracting features: {i} / {len(loader)}", end="")

        print()

        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        return features, labels

    def run_tsne(self, perplexity=30, random_state=42):
        features, labels = self.extract_features()

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state
        )
        feat_2d = tsne.fit_transform(features)
        return feat_2d, labels

    def plot(self, perplexity=30, random_state=42, save_path=None):
        feat_2d, labels = self.run_tsne(
            perplexity=perplexity,
            random_state=random_state
        )

        plt.figure(figsize=(8, 6))
        for c in np.unique(labels):
            idx = labels == c
            plt.scatter(
                feat_2d[idx, 0],
                feat_2d[idx, 1],
                label=f'Class {c}',
                s=10
            )

        plt.legend()
        plt.title(f't-SNE of projected features W_Neg=100')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            print(f"Saved t-SNE plot to: {save_path}")

        plt.show()