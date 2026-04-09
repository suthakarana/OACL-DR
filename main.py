import sys
import torch
import argparse
import time
import numpy as np
from Models.CNN_Train import *
from Models.logger import *
from datetime import timedelta
from Models.EvaluateEnsemble import *
from Models.TSNEPlot import *
from Models.Tsne_centroidPlot import *
from Models.confusion_qualitative import *
from Models.Heatmap import *

desc =  'DenseNet_ENS_'
dataset =  'DR'          #'LIMUC' 
modelName = 'densenet121'  #'resnet18' 
Prototype_mode = 'Saved' #'Calculated'

gpuid = 0

useBothEyes = True
projDim = 512
mocoK = 512

if dataset == 'DR':
    if modelName == 'densenet121':
        if useBothEyes:
            bs_Tr = 10
        else:
            bs_Tr = 16
    else:
        bs_Tr = 64
else:
    bs_Tr = 64
bs_Te = 128
bs_Val = 128

parser = argparse.ArgumentParser()
parser.add_argument("--desc", default=desc)
parser.add_argument("--dataset", default=dataset)
parser.add_argument("--bs_Tr", type=int, default=bs_Tr)
parser.add_argument("--bs_Te", type=int, default=bs_Te)
parser.add_argument("--bs_Val", type=int, default=bs_Val)
parser.add_argument("--modelName", default=modelName)
parser.add_argument("--n_epochs", type=int, default= 30)
parser.add_argument("--gpuid", type=int, default=gpuid)
parser.add_argument("--useBothEyes", default=useBothEyes)
parser.add_argument("--projDim", type=int, default=projDim)
parser.add_argument("--mocoK", default=mocoK)
parser.add_argument("--lr", default= 1e-3)
parser.add_argument("--pL", default= 1.0)
parser.add_argument("--w_mse", default= 1)
parser.add_argument("--w_ce", default= 1)
parser.add_argument("--w_fea", default= 1)
parser.add_argument("--w_proto", default=0)
parser.add_argument("--w_neg", default=100)
parser.add_argument("--weight_decay", default= 5e-4)
parser.add_argument("--alpha_ordinal", default=2)
parser.add_argument("--useCW", default= 0)
parser.add_argument("--T", default=0.1)
parser.add_argument("--Prototype_mode", default=Prototype_mode)


opt = parser.parse_args()
dirname = os.path.join('/home/suthakaran/Codes/DR_softMAX/Results/', modelName)
if not os.path.exists(dirname):
    os.makedirs(dirname)
fn = os.path.join(dirname, desc)

print(fn)
sys.stdout = Logger(fn)


def printResults(paraArr, meanArr, stdArr):
    paraArr = np.stack(paraArr, axis=0)
    meanArr = np.stack(meanArr, axis=0)
    stdArr = np.stack(stdArr, axis=0)
    print(150 * '-')
    for r in range(len(paraArr)):
        para = paraArr[r]
        for e in para:
            print(e, ':', end='')
        print(end='|')

        mean_val = meanArr[r]
        std_val = stdArr[r]

        for i in range(len(mean_val)):
            print(f"{mean_val[i]:.4f} ± {std_val[i]:.4f}", end=' | ')
        print()
    print(150 * '-')

# opt.seed = 100
# EvaluateEnsemble(opt)
# exit()

# opt.seed = 1000
# path = '/home/suthakaran/Codes/DR_softMAX/TrainedModels/order_only_models/model0.pt'
# tsne_plotter = TSNEPlot(opt, path, split='test')
# tsne_plotter.plot(perplexity=30, save_path='tsne_model0_test.png')
# exit()


re = []
para = []
mean_re = []
std_re = []

opt.w_pos = 0

for w_mse in [1]:
    opt.w_mse = w_mse

    for w_neg in [100]:
        opt.w_neg = w_neg

        tmp_re_list = []

        for i in range(2,3):
            print(opt)
            opt.seed = (i + 1) * 1000
            print(f"\n--- Model {i + 1} | Seed {opt.seed} ---")

            cnn = CNN_Train(opt)
            tmp_re, ema_model_state, prototypes = cnn.iterate()

            tmp_re = np.array(tmp_re, dtype=np.float32)
            print(
                f"{tmp_re[0]:.4f}\t{tmp_re[1]:.4f}\t{tmp_re[2]:.4f}\t{tmp_re[3]:.4f} | "
                f"{tmp_re[4]:.4f}\t{tmp_re[5]:.4f}\t{tmp_re[6]:.4f}\t{tmp_re[7]:.4f} | "
                f"{tmp_re[8]:.4f}\t{tmp_re[9]:.4f}\t{tmp_re[10]:.4f}\t{tmp_re[11]:.4f}"
            )

            path = f'/home/suthakaran/Codes/DR_softMAX/TrainedModels/{opt.desc}_model{i}.pt'
            torch.save({
                'model_state_dict': ema_model_state,
                'prototypes': prototypes.cpu(),
                'results': tmp_re,
                'seed': opt.seed
            }, path)

            tmp_re_list.append(tmp_re)

        tmp_re_list = np.stack(tmp_re_list, axis=0)
        mean_result = np.mean(tmp_re_list, axis=0)
        std_result = np.std(tmp_re_list, axis=0)
        re.append(tmp_re_list.tolist())
        mean_re.append(mean_result.tolist())
        std_re.append(std_result.tolist())

        para.append([opt.pL, opt.lr, opt.alpha_ordinal, opt.w_neg])

        def fmt_block(start):
            return "  ".join([
                f"{mean_result[i]:.2f}±{std_result[i]:.2f}" if i % 4 < 2
                else f"{mean_result[i]:.3f}±{std_result[i]:.3f}"
                for i in range(start, start + 4)
            ])

        print(f"{fmt_block(0)} | {fmt_block(4)} | {fmt_block(8)}")

        printResults(para, mean_re, std_re)