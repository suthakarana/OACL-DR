import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import torchutils as tu
import torch.optim as optimfrom Models.util import *
from Models.DataLoader import *
from Models.ResNet import *
from Models.DenseNet import *
from Models.QKappa import quadratic_weighted_kappa
from Models.OptimizedRounder import *
from torch.autograd import Variable
from Models.loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch_ema import ExponentialMovingAverage
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from pytorch_lightning import seed_everything
from Models.MoCoFixed import *
from oprounder import OptimizedRounder
from Models.CNN_Train import *

class EvaluateEnsemble(nn.Module):
    def __init__(self, opt):
        super(EvaluateEnsemble, self).__init__()
        self.opt = opt
        self.opt.bs_Tr = 128

        self.trainLoader, self.testLoader, self.valLoader, self.uniqueLbls \
            = getDataLoaders(opt.dataset, opt.pL, opt.seed, opt.bs_Tr, opt.bs_Te, opt.bs_Val)
        self.nClass = len(self.uniqueLbls)
        self.sm = nn.Softmax(dim=1)
        self.optimized_rounder_MSE = OptimizedRounder(n_classes=self.nClass, metric='quadratic_kappa')
        self.criterion_ORDER = CDWloss(self.nClass, self.opt.alpha_ordinal, self.opt.T, self.opt.w_neg)
        self.nmodels = 5

        self.net = []
        self.prototypes_list = []
        self.protoLbls_list = []

        print('----- Loading models -----')

        for i in range(self.nmodels):
            path = '/home/suthakaran/Codes/DR_softMAX/TrainedModels/Test_ENS1_model' + str(i) + '.pt'
            ckpt = torch.load(path, map_location='cpu')
            if self.opt.modelName == 'densenet121':
                model = MyDenseNet(num_classes=self.nClass,
                                   useBothEyes=self.opt.useBothEyes,
                                   proj_dim=self.opt.projDim)
            else:
                model = MyResNet(num_classes=self.nClass,
                                 useBothEyes=self.opt.useBothEyes,
                                 proj_dim=self.opt.projDim)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.cuda(self.opt.gpuid)
            model.eval()
            print('-----')
            self.net.append(model)
            
            if self.opt.Prototype_mode =='Saved':
                prototypes = ckpt['prototypes'].cuda(self.opt.gpuid)
                self.prototypes_list.append(prototypes)
                print(f'Model {i} loaded with saved prototypes')

        if self.opt.Prototype_mode == 'Calculated':
            print('----- Computing per-model prototypes -----')
            for i in range(self.nmodels):
                prototypes, protoLbls = self.CalPrototypes_single_model(i)
                self.prototypes_list.append(prototypes)
                self.protoLbls_list.append(protoLbls)
                print(f'Prototypes computed for model {i}')

        self.test(self.testLoader)

    def printProgress(self, message, i, tot):
        print(f"\r{message} {i} of {tot}:", end="")

    @torch.no_grad()
    def CalPrototypes_single_model(self, model_idx):
        model = self.net[model_idx].cuda(self.opt.gpuid)
        model.eval()

        zArr = []
        yArr = []

        for i, (I1, I2, y, _) in enumerate(self.trainLoader):
            I1 = I1.cuda(self.opt.gpuid, non_blocking=True)
            I2 = I2.cuda(self.opt.gpuid, non_blocking=True)
            y = y.cuda(self.opt.gpuid, non_blocking=True)

            z, _, _ = model(I1, I2)
            zArr.append(z)
            yArr.append(y)
            self.printProgress(f'Prototypes Model {model_idx}: ', i, len(self.trainLoader))

        zArr = torch.cat(zArr, dim=0)
        yArr = torch.cat(yArr, dim=0)

        prototypes = torch.zeros(self.nClass, zArr.shape[1], device=zArr.device)
        for c in range(self.nClass):
            idx = (yArr == c)
            if idx.sum() > 0:
                prototypes[c] = zArr[idx].mean(0)

        return prototypes, torch.arange(self.nClass, device=zArr.device)

    def testImage(self, I1, I2):
        I1 = I1.cuda(self.opt.gpuid, non_blocking=True)
        I2 = I2.cuda(self.opt.gpuid, non_blocking=True)

        o_mse_arr = []
        o_ce_logits_arr = []
        prob_prot_arr = []

        for i in range(self.nmodels):
            with torch.no_grad():
                z, o_mse, o_ce = self.net[i](I1, I2)

            o_mse_arr.append(o_mse)
            o_ce_logits_arr.append(o_ce)

            prob_prot = self.criterion_ORDER.getProbClassInference( z, self.prototypes_list[i])
            prob_prot_arr.append(prob_prot)

        o_mse = torch.stack(o_mse_arr, dim=2)  
        o_ce_logits = torch.stack(o_ce_logits_arr, dim=2)  
        prob_prot = torch.stack(prob_prot_arr, dim=2)  

        return o_mse, o_ce_logits, prob_prot

    @torch.no_grad()
    def test(self, loader):
        print('\n----- Starting ensemble evaluation -----')

        pred_all_mse = []
        pred_all_ce = []
        prob_prot_all = []
        gt_all = []

        for i, (I1, I2, y, _) in enumerate(loader):
            I1 = I1.cuda(self.opt.gpuid)
            I2 = I2.cuda(self.opt.gpuid)
            y = y.cuda(self.opt.gpuid)

            o_mse, o_ce_logits, prob_prot = self.testImage(I1, I2)

            pred_all_mse.append(o_mse)             
            pred_all_ce.append(o_ce_logits)        
            prob_prot_all.append(prob_prot)        
            gt_all.append(y)

            self.printProgress('Testing: ', i, len(loader))

        print()

        gt_all = torch.cat(gt_all, 0)                
        mse_all = torch.cat(pred_all_mse, 0)         
        ce_logits_all = torch.cat(pred_all_ce, 0)    
        prob_prot_all = torch.cat(prob_prot_all, 0)  

        gt_np = gt_all.view(-1, 1).float().cpu().numpy()

        mse_mean = mse_all.mean(dim=2)                
        ce_mean = self.sm(ce_logits_all.mean(dim=2))    
        prob_prot_mean = prob_prot_all.mean(dim=2)    

        for i in range(self.nmodels + 1):
            if i < self.nmodels:
                mse_i = mse_all[:, :, i]                 
                ce_i = self.sm(ce_logits_all[:, :, i])   
                order_i = prob_prot_all[:, :, i]         
            else:
                mse_i = mse_mean                         
                ce_i = ce_mean                      
                order_i = prob_prot_mean                 

            self.optimized_rounder_MSE.fit(mse_i.cpu().numpy(), gt_np)
            mse_rounded = self.optimized_rounder_MSE.predict(mse_i.cpu().numpy())

            reTe = {
                'MSE': getScores(gt_np, mse_rounded),
                'CE': getScore_fromPred(gt_np, ce_i),
                'ORDER': getScore_fromPred(gt_np, order_i),
            }

            print('Model : ', i)
            self.printScore(reTe['MSE'], '\t|\t')
            self.printScore(reTe['CE'], '\t|\t')
            self.printScore(reTe['ORDER'], '\t|\t\n')


        return gt_np, mse_all, ce_logits_all, prob_prot_all

    def printScore(self, re, ends='\n'):
        print('%2.4f\t %2.4f\t%.4f\t%.4f' % (re[0], re[1], re[2], re[3]), end=ends)


