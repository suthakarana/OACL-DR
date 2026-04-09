import csv
import os
import numpy as np
import math
import torch
import sys
import randomimport torch.utils.data as data
import scipy.iofrom PIL import Image
import torchvision.transforms as transforms
from PIL import ImageChops
from sklearn.utils import shuffle

def getData_DR(fn, srcDir, train):
    lblarr = []
    imgFnarr = []
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        if line[0] == 'image': continue
        lbl = int(line[1])
        if len(line) > 2:
            tmp = line[2].strip()
        if (train == 'TRAIN') or ((train == 'VAL') and (tmp == 'Public')) or ((train == 'TEST') and (tmp == 'Private')):
            lblarr.append(lbl)
            imgFnarr.append(os.path.join(srcDir, line[0] + '.png'))
    file.close()
    return imgFnarr, lblarr

def getData_LIMUC(srcDir):
    lblarr = []
    imgFnarr = []
    classes = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']

    for i, class_name in enumerate(classes):
        subdir = os.path.join(srcDir, class_name)
        fnArr = os.listdir(subdir)
        for fn in fnArr:
            imgFnarr.append(os.path.join(subdir, fn))
            lblarr.append(i)
    imgFnarr = np.array(imgFnarr)
    lblarr = np.array(lblarr)
    return imgFnarr, lblarr


def splitData_Pecentage(lblArr, pL,seed):
    uniqueLbls = np.unique(lblArr)
    trainL_idx = []
    trainUL_idx = []
    for lbl in uniqueLbls:
        idx = [i for i, x in enumerate(lblArr) if x==lbl]
        idx = shuffle(idx, random_state=seed)
       
        nl = int(np.ceil(len(idx) * pL))
        tmpL = idx[0:nl]
        trainL_idx.extend(tmpL)

        if pL != 1:
            tmpUL = idx[nl:]
            trainUL_idx.extend(tmpUL)
    return trainL_idx, trainUL_idx


def cal_priors__(lbl_arr):
    unique_lbls, counts = np.unique(lbl_arr, return_counts=True)
    total_samples = len(lbl_arr)
    priors = counts / total_samples
    priors = np.array(priors, dtype=np.float32)
    priors = torch.from_numpy(priors)
    print("Priors :", priors ,end='')
    print('\n-----------------\n')
    return priors

def getIdx(lbl_arr, lbl_search):
    return [i for i, x in enumerate(lbl_arr) if x == lbl_search]

def calWeights(lbl_arr):
    weights = []
    unique_lbls = np.unique(lbl_arr)
    print('Total images : ', len(lbl_arr))
    for lbl in unique_lbls:
        idx = getIdx(lbl_arr, lbl)
        weights.append(1/len(idx))
        print(lbl, ' : ', len(idx))
    weights = np.asarray(weights)
    weights = weights / weights.sum()
    print("Weights :", end='')
    for val in weights:
       print(np.round(val, 3), end=',')
    print('\n-----------------\n')
    return weights

def calWeights_oneVsOthers(lbl_arr):
    weights = []
    unique_lbls = np.unique(lbl_arr)
    total_samples = len(lbl_arr)
    print('Total images:', total_samples)
    for lbl in unique_lbls:
        idx = getIdx(lbl_arr, lbl)
        class_count = len(idx)
        other_count = total_samples - class_count
        weight = (1 / class_count) / ((1 / other_count) +(1/class_count))
        weights.append(weight)
        print(lbl, ' : ', len(idx))
    weights = np.asarray(weights)
    for val in weights:
        print(f" {np.round(val, 3)}", end=',')
    print('-----------------\n')

    return weights


def getTransform(dataset):
    if dataset == 'DR':
        imsize =  512
        imsize2 = 448
    else:
        imsize = 256
        imsize2 = 224

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.08, 0.08, 0.08])

    transform_train_W = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_train_S = transforms.Compose([
        transforms.RandomAffine([0, 360], translate=[0.1, 0.1], scale=[0.8, 1.2], shear=20),  # translate=[0.02, 0.02],
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform_train_W, transform_train_S, transform_test

def printStatDatasets(dataset_L, dataset_UL, dataset_Te, dataset_Val):
    y_L, y_te, y_val = dataset_L.lblArr, dataset_Te.lblArr, dataset_Val.lblArr
    y_UL = None
    if dataset_UL is not None:
        y_UL = dataset_UL.lblArr
    unique_lbls = np.unique(y_L)

    def countLbls(lblArr, lbl):
        idx = [i for i, x in enumerate(lblArr) if x == lbl]
        return len(idx)

    print('lbl \t Lbl \t UNLbl \t Test \t Validation')
    for lbl in unique_lbls:
        tmp_L = countLbls(y_L, lbl)
        tmp_un = 0
        if y_UL is not None:
            tmp_un = countLbls(y_UL, lbl)
        tmp_te = countLbls(y_te, lbl)
        tmp_val = countLbls(y_val, lbl)
        print('%1d\t%5d\t%5d\t%5d\t%5d'%(lbl, tmp_L, tmp_un, tmp_te, tmp_val))
    if y_UL is not None:
        print('\t%5d\t%5d\t%5d\t%5d' % (len(y_L), len(y_UL), len(y_te), len(y_val)))
    else:
        print('\t%5d\t%5d\t%5d\t%5d' % (len(y_L), 0, len(y_te),len(y_val)))


class DataSetDR(data.Dataset):
    def __init__(self, dataset, fnArr, lblArr, train='TRAIN'):
        self.transformTrain_W, self.transformTrain_S, self.transformTest = getTransform(dataset)
        self.train = train
        self.fnArr = fnArr
        self.lblArr = lblArr

    def __getitem__(self, index):
        img_fn = self.fnArr[index]
        target = self.lblArr[index]
        I1 = pil_loader(img_fn)

        if str.find(img_fn, 'right') > 0:
            img_fn2 = img_fn.replace('right', 'left')
        else:
            img_fn2 = img_fn.replace('left', 'right')

        I2 = pil_loader(img_fn2)

        if self.train in ['TEST', 'VAL']:
            return self.transformTest(I1), self.transformTest(I2), target, img_fn
        else:
            return self.transformTrain_S(I1), self.transformTrain_S(I2), target, img_fn

    def __len__(self):
        return len(self.lblArr)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def get_datasetsDR(pL, seed):
    src_dir = '/home/suthakaran/Data/DR/'

    img_dir = os.path.join(src_dir, 'Preprocessed_512/Train/')
    annot_fn = os.path.join(src_dir, 'trainLabels.csv')
    fnArr_tr, lbl_tr = getData_DR(annot_fn, img_dir, 'TRAIN')

    fnArr_tr = np.array(fnArr_tr)
    lbl_tr = np.array(lbl_tr)

    trainL_idx, trainUL_idx = splitData_Pecentage(lbl_tr, pL, seed)

    xtr = fnArr_tr[trainL_idx].tolist()
    ytr = lbl_tr[trainL_idx].tolist()

    # TEST
    img_dir = os.path.join(src_dir, 'Preprocessed_512/Test/')
    annot_fn = os.path.join(src_dir, 'retinopathy_solution.csv')
    xte, yte = getData_DR(annot_fn, img_dir, 'TEST')

    # VALIDATION
    xv, yv = getData_DR(annot_fn, img_dir, 'VAL')

    return xtr, ytr, xv, yv, xte, yte


def get_datasetsLIMUC():
    srcDir = '/home/suthakaran/Data/LIMUC'
    xtr, ytr = getData_LIMUC(os.path.join(srcDir, 'train_and_validation_sets'))
    xv, yv = getData_LIMUC(os.path.join(srcDir, 'train_and_validation_sets'))
    xte, yte = getData_LIMUC(os.path.join(srcDir, 'test_set'))
    return xtr, ytr, xv, yv, xte, yte

def getDataLoaders(dataset, pL, seed, bs_Tr, bs_Te, bs_Val):
    if dataset == 'DR':
        xtr, ytr, xv, yv, xte, yte = get_datasetsDR(pL, seed)
    else:
        xtr, ytr, xv, yv, xte, yte = get_datasetsLIMUC()
    ul = np.unique(ytr)

    trainset_L = DataSetDR(dataset, xtr, ytr, 'TRAIN')
    trainloader_L = torch.utils.data.DataLoader(trainset_L,
                                                batch_size=bs_Tr,
                                                num_workers=12,
                                                shuffle=True)

    testset = DataSetDR(dataset, xte, yte, 'TEST')
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs_Te,
                                             num_workers=4,
                                             shuffle=False)

    valset = DataSetDR(dataset, xv, yv, 'VAL')
    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size= bs_Val,
                                             num_workers=16,
                                             shuffle=False)


    printStatDatasets(trainset_L, None, testset, valset )
    return trainloader_L, testloader, valloader,  ul
