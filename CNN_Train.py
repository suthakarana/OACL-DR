import copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os, time
import torchutils as tu
from Models.DenseNet import *
from Models.ResNet import *
from oprounder import OptimizedRounder
from torch.autograd import Variable
from Models.loss import *
from torch_ema import ExponentialMovingAverage
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from pytorch_lightning import seed_everything
from Models.MoCoFixed import *
from Models.CDWLoss import *
from Models.DataLoader import *
from Models.util import *
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, _LRScheduler, SequentialLR, LinearLR

class CNN_Train(nn.Module):
    def __init__(self, opt, inference_only=False):
        super(CNN_Train, self).__init__()
        self.moco = None
        seed_everything(opt.seed)
        self.opt = opt
        self.trainLoader, self.testLoader, self.valLoader, self.uniqueLbls \
            = getDataLoaders(opt.dataset, opt.pL, opt.seed, opt.bs_Tr, opt.bs_Te, opt.bs_Val)

        self.nClass = len(self.uniqueLbls)
        self.sm = nn.Softmax(dim=1)
        self.initialize_network()
        self.initMoco()

    def initialize_network(self):
        if self.opt.modelName == 'densenet121':
            net = MyDenseNet(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes, proj_dim=self.opt.projDim)
        else:
            net = MyResNet(num_classes=self.nClass, useBothEyes=self.opt.useBothEyes, proj_dim=self.opt.projDim)
        self.net = net.cuda(self.opt.gpuid)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.999)
        if not self.opt.useCW:
            self.cw = None
        else:
            cw = calWeights(self.trainLoader.dataset.lblArr)
            self.cw = torch.FloatTensor(cw).cuda(self.opt.gpuid)

        self.optimized_rounder_MSE = OptimizedRounder(n_classes=self.nClass,  metric='quadratic_kappa')
        self.optimized_rounder_CE = OptimizedRounder(n_classes=self.nClass,  metric='quadratic_kappa')
        self.optimized_rounder_ORDER = OptimizedRounder(n_classes=self.nClass,  metric='quadratic_kappa')

        self.criterion_mse = nn.MSELoss().cuda(self.opt.gpuid)
        self.criterion_ce = CE(self.nClass, self.opt.gpuid, self.cw).cuda(self.opt.gpuid)

        self.moco = MoCoFixed(num_classes=self.nClass, dim=self.opt.projDim, K=self.opt.mocoK).cuda(self.opt.gpuid)

        self.criterion_ORDER = CDWloss(self.nClass, self.opt.alpha_ordinal,self.opt.gpuid,  self.opt.w_mse, self.opt.T, self.opt.w_neg)

        self.optimizer = optim.SGD([
             {"params": self.net.features.parameters(),          "weight_decay": 0.2* self.opt.weight_decay},
             {"params": self.net.projection_head.parameters(),   "weight_decay": self.opt.weight_decay},
             {"params": self.net.fc_mse.parameters(),            "weight_decay": self.opt.weight_decay},
             {"params": self.net.fc_ce.parameters(),             "weight_decay": self.opt.weight_decay},
         ], lr=self.opt.lr, momentum=0.9, nesterov=True)

        steps_per_epoch = len(self.trainLoader)
        total_steps = self.opt.n_epochs * steps_per_epoch
        warmup_epochs = 3
        warmup_steps = warmup_epochs * steps_per_epoch

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )

    @torch.no_grad()
    def CalPrototypes(self):
        fea_moco, y_moco = self.moco.getDict()
        centroids = []
        lbls = []
        for c in range(self.nClass):
            idx = (y_moco == c).view(-1)
            if idx.sum() > 0:
                c_fea = fea_moco[idx].mean(0, keepdim=True)
                centroids.append(c_fea)
            else:
                centroids.append(torch.full((1, fea_moco.shape[1]), -1e9, device=fea_moco.device))
            lbls.append(c)
        centroids = torch.cat(centroids, dim=0)
        lbls = torch.from_numpy(np.array(lbls)).view(-1).cuda(self.opt.gpuid)
        return centroids, lbls

    def getSamplesPerClass(self):
        yall = torch.tensor(self.trainLoader.dataset.lblArr).cuda(self.opt.gpuid)
        samples_per_class = []
        for c in range(self.nClass):
            n = torch.sum(yall == c)
            samples_per_class.append(n)
        samples_per_class = torch.stack(samples_per_class)
        return samples_per_class

    def getLoss(self, out_mse, out_ce, fea, y,  prototypes):
        device = y.device
        loss_ce = torch.tensor(0., device=device)
        loss_mse = torch.tensor(0., device=device)
        loss_fea = torch.tensor(0., device=device)

        if self.opt.w_ce > 0 :
            loss_ce = self.opt.w_ce * self.criterion_ce.CE(out_ce, y)
        if self.opt.w_mse > 0:
            loss_mse = self.opt.w_mse * self.criterion_mse(out_mse, y.view(-1, 1).float())
        if self.opt.w_fea > 0:
            fea_moco, y_moco = self.moco.getDict()
            loss_fea = self.opt.w_fea * self.criterion_ORDER(fea, y, fea_moco, y_moco,  prototypes)

        loss = loss_ce + loss_fea  + loss_mse
        return loss, loss_mse, loss_ce, loss_fea

    @torch.no_grad()
    def initMoco(self):
        tot = 2
        for k in range(tot):
            for i, (I1, I2, y, _) in enumerate(self.trainLoader):
                I1 = I1.to(self.opt.gpuid, non_blocking=True)
                I2 = I2.to(self.opt.gpuid, non_blocking=True)
                y = y.to(self.opt.gpuid, non_blocking=True)
                z = self.testImage(I1, I2, None)
                self.moco.dequeue_and_enqueue(z.detach(), y)
                self.printProgress('init moco :', k*len(self.trainLoader) + i, tot * len(self.trainLoader))
        print()

    def train(self, epoch):
        self.net.train()

        pred_all_mse, pred_all_ce, gt_all, prob_prot_all = [], [], [], []
        loss_tot, loss_tot_mse, loss_tot_ce, loss_tot_fea = 0,0,0,0
        for i, (I1, I2, y, _) in enumerate(self.trainLoader):
            I1 = I1.to(self.opt.gpuid, non_blocking=True)
            I2 = I2.to(self.opt.gpuid, non_blocking=True)
            y = y.to(self.opt.gpuid, non_blocking=True)

            if i % 50 == 0:
                prototypes, protoLbls = self.CalPrototypes()

            z,  out_mse, out_ce =  self.net(I1, I2)
            loss, loss_mse, loss_ce, loss_fea = self.getLoss(out_mse, out_ce, z, y,  prototypes)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update()

            loss_tot += loss.item()
            loss_tot_mse += loss_mse.item()
            loss_tot_ce += loss_ce.item()
            loss_tot_fea += loss_fea.item()

            prob_prot = self.criterion_ORDER.getProbClassInference(z.detach(), prototypes)
            gt_all.append(y.detach())
            pred_all_mse.append(out_mse.detach())
            pred_all_ce.append(out_ce.detach())
            prob_prot_all.append(prob_prot.detach())
            self.printProgress('Train :', i, len(self.trainLoader))

            zq = self.testImage(I1, I2, None)
            self.moco.dequeue_and_enqueue(zq.detach(), y)

        gt_all = torch.cat(gt_all, 0)
        pred_all_mse = torch.cat(pred_all_mse, 0)
        pred_all_ce = self.sm(torch.cat(pred_all_ce, 0))
        prob_prot_all = torch.cat(prob_prot_all, 0)

        if epoch == 0:
            self.optimized_rounder_MSE.fit(pred_all_mse.cpu().numpy(), gt_all.view(-1, 1).float().cpu().numpy())

            cal_ce = getLinear(pred_all_ce)
            self.optimized_rounder_CE.fit(cal_ce.cpu().numpy(), gt_all.view(-1, 1).float().cpu().numpy())

            cal_order = getLinear(prob_prot_all)
            self.optimized_rounder_ORDER.fit(cal_order.cpu().numpy(), gt_all.view(-1, 1).float().cpu().numpy())

        pl_MSE = self.optimized_rounder_MSE.predict(pred_all_mse.cpu().numpy())
        re_mse = getScores(gt_all.cpu().numpy(), pl_MSE)
        re_ce = getScore_fromPred(gt_all.cpu().numpy(), pred_all_ce)
        re_order = getScore_fromPred(gt_all.cpu().numpy(), prob_prot_all)

        _, pl = torch.max(prob_prot_all, dim=1)
        cm = confusion_matrix(gt_all.cpu().numpy(), pl.cpu().numpy())
        self.printCM(cm)

        return  loss_tot, loss_tot_mse, loss_tot_ce, loss_tot_fea, re_mse[2], re_ce[2], re_order[2]

    def LearnOptimizedRounder(self):
        print('fitting optimized rounder..')
        _, vals = self.test(self.valLoader)
        gt_all = vals['gt_all'].view(-1, 1).float().cpu().numpy()
        self.optimized_rounder_MSE.fit(vals['pred_all_mse'].cpu().numpy(), gt_all)

        cal_ce = getLinear(vals['pred_all_ce']).cpu().numpy()
        self.optimized_rounder_CE.fit(cal_ce, gt_all)

        cal_order = getLinear(vals['prob_prot_all']).cpu().numpy()
        self.optimized_rounder_ORDER.fit(cal_order, gt_all)
        print('done')

    def printCM(self, cm):
        print()
        for r in range(self.nClass):
            for c in range(self.nClass):
                print(cm[r,c], end='\t')
            print()

    def printProgress(self, message, i, tot):
        print(f"\r{message} {i} of {tot}:", end="")

    @torch.no_grad()
    def testImage(self, I1, I2,  prototypes=None, protoLbls=None):
        with torch.no_grad():
            with self.ema.average_parameters():
                z, o_mse, o_ce = self.net(I1, I2)
            if prototypes is not None:
                prob_prot = self.criterion_ORDER.getProbClassInference(z, prototypes)
        if prototypes is not None:
            return z, o_mse, o_ce, prob_prot
        else:
            return z

    @torch.no_grad()
    def test(self, loader):
        prototypes, protoLbls = self.CalPrototypes()

        self.net.eval()
        pred_all_mse, pred_all_ce, gt_all, prob_prot_all = [], [], [], []
        for i, (I1, I2, y, _) in enumerate(loader):
            I1 = I1.to(self.opt.gpuid, non_blocking=True)
            I2 = I2.to(self.opt.gpuid, non_blocking=True)
            y = y.to(self.opt.gpuid, non_blocking=True)

            _, out_mse, out_ce, prob_prot = self.testImage(I1, I2, prototypes, protoLbls)
            gt_all.append(y.detach())
            pred_all_mse.append(out_mse.detach())
            pred_all_ce.append(out_ce.detach())
            prob_prot_all.append(prob_prot.detach())
            self.printProgress('Test :', i, len(loader))
        gt_all = torch.cat(gt_all, 0)
        pred_all_mse = torch.cat(pred_all_mse, 0)
        pred_all_ce = self.sm(torch.cat(pred_all_ce, 0))
        prob_prot_all = torch.cat(prob_prot_all, 0)
        pred_all_mse_rounded = self.optimized_rounder_MSE.predict(pred_all_mse.cpu().numpy())
        cal_ce = getLinear(pred_all_ce)
        pred_cal_ce_rounded = self.optimized_rounder_CE.predict(cal_ce.cpu().numpy())
        cal_order = getLinear(prob_prot_all)
        pred_cal_order_rounded = self.optimized_rounder_ORDER.predict(cal_order.cpu().numpy())
        re = {'MSE':getScores(gt_all.cpu().numpy(), pred_all_mse_rounded),
              'CE_C': getScores(gt_all.cpu().numpy(), pred_cal_ce_rounded),
              'ORDER_C': getScores(gt_all.cpu().numpy(), pred_cal_order_rounded),
              'CE': getScore_fromPred(gt_all.cpu().numpy(), pred_all_ce),
              'ORDER': getScore_fromPred(gt_all.cpu().numpy(), prob_prot_all),
        }
        vals = {
            'gt_all':gt_all,
            'pred_all_mse':pred_all_mse,
            'pred_all_ce':pred_all_ce,
            'prob_prot_all':prob_prot_all
        }
        _, pl = torch.max(prob_prot_all, dim=1)
        cm = confusion_matrix(gt_all.cpu().numpy(), pl.cpu().numpy())
        self.printCM(cm)

        return re, vals

    def printScore(self, re, ends):
        print('%2.2f\t %2.2f\t%.3f\t%.3f' % (re[0], re[1], re[2], re[3]), end=ends)

    def iterate(self):
        print('\nEpoch\tlr\tTrLoss\tTrK\t|TeLoss\tTeK\t\tTeMCA\t | Te_F1\t ')
        print('-' * 85)
        results = []
        start_time = time.time()
        reTe = 0
        for epoch in range(self.opt.n_epochs):
            print('\nEpoch: ', epoch)
            lr = tu.get_lr(self.optimizer)
            loss_tot, loss_tot_mse, loss_tot_ce, loss_tot_fea, k_mse, k_ce, k_order = self.train(epoch)
            if epoch % 5 == 0 or epoch > self.opt.n_epochs - 6:
                self.LearnOptimizedRounder()
                reTe, _ = self.test(self.testLoader)

            re = [lr, loss_tot, loss_tot_mse, loss_tot_ce, loss_tot_fea, k_mse, k_ce, k_order]
            re.extend(reTe['MSE'])
            re.extend(reTe['CE'])
            re.extend(reTe['ORDER'])

            results.append(re)
            print('%5d\t %5.3f\t %5.3f\t %5.5f\t  %5.5f\t %.3f\t%.3f\t%.3f' % (epoch, loss_tot, loss_tot_mse, loss_tot_ce, loss_tot_fea, k_mse, k_ce, k_order))

            print('\n')
            self.printScore(reTe['MSE'], '\n')
            self.printScore(reTe['CE'], '\t|\t')
            self.printScore(reTe['ORDER'], '\t|\t\n')

        results = np.array(results)
        results = results[:, -3*4:]
        results = results[-5:, :]
        results = np.round(np.mean(results, axis=0), 4)

        with self.ema.average_parameters():
            prototypes, _ = self.CalPrototypes()
            ema_model_state = copy.deepcopy(self.net.state_dict())

        print("--- %s seconds ---" % (time.time() - start_time),'\n' )
        return results, ema_model_state , prototypes