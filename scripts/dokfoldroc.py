import logging
import os
import sys
import math
import argparse
sys.path.append('scripts')
sys.path.append('share')
sys.path.append('python')
from collections import namedtuple
import numpy as np
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

def doNumber(data_EC, data_Layer, data_number, data_true):

    data = {}

    outdata = np.zeros(
        (data_number.shape[0],),
        dtype=[
            ('Output_number', np.dtype(('f4', 3))),
            ('Output_number_true', 'i4')
        ]
    )

    IBL = np.logical_and(data_Layer == 0, data_EC == 0)
    Barrel = np.logical_and(data_Layer > 0, data_EC == 0)
    Endcap = data_EC != 0

    outdata['Output_number'] = data_number
    outdata['Output_number_true'] = data_true

    data['IBL'] = outdata[IBL]
    data['Barrel'] = outdata[Barrel]
    data['Endcap'] = outdata[Endcap]
    return data

def rocGraph(data, classes, name, folds):

    fpr = [[0]*(folds) for i in range(len(data))]
    tpr = [[0]*(folds) for i in range(len(data))]
    auc1 = [[0]*(folds) for i in range(len(data))]

    npoints = 200
    base_fpr = np.exp(np.linspace(math.log(0.0005), 0., npoints))
    colors = ["r", "g", "b"]

    pos, neg = classes
    linetypes = ['-', '--', ':']

    for j, layer in enumerate(data):

        for i, data_split in enumerate(data[layer]):
        
            pos_sel = data_split['Output_number_true'] == pos
            neg_sel = data_split['Output_number_true'] == neg

            isel = np.where(
                np.logical_or(
                    pos_sel,
                    neg_sel,
                )
            )[0]

            fpr[j][i], tpr[j][i], _ = roc_curve(
                data_split['Output_number_true'][isel],
                data_split['Output_number'][isel][:, pos - 1],
                pos_label=pos
            )
            auc1[j][i] = auc(fpr[j][i], tpr[j][i])
        
    for i, layer in enumerate(data):
        
        tpr_array = np.array([])

        for j in range(folds):
            tpr_interpolated = np.interp(base_fpr, fpr[i][j], tpr[i][j])
            tpr_interpolated = tpr_interpolated.reshape((1,npoints))
            tpr_array = np.concatenate([tpr_array, tpr_interpolated], axis=0) if tpr_array.size else tpr_interpolated
            
        mean_tpr = np.mean(tpr_array, axis=0)
        rms_tpr = np.std(tpr_array, axis=0)
        plus_tpr = np.minimum(mean_tpr+rms_tpr, np.ones(npoints))
        minus_tpr = np.maximum(mean_tpr-rms_tpr,np.zeros(npoints))
        plt.plot(base_fpr, mean_tpr, linetypes[i], label=f'{layer} (AUC = {np.mean(auc1[i]):.2f} (+- {np.std(auc1[i]):.4f}))')
        plt.fill_between(base_fpr, minus_tpr, plus_tpr, alpha=0.3)
        
        
    rand_chance = np.linspace(0, 1, 100)
    plt.plot(rand_chance,rand_chance, ':', color='grey', label='Random (AUC = 0.5)')
    plt.semilogx()
    plt.ylabel(f"Pr(Estimated: {pos}-particle | True: {pos}-particle)")
    plt.xlabel(f"Pr(Estimated: {pos}-particle | True: {neg}-particle)")
    plt.xlim([0.001, 1.05])
    plt.ylim(0,1.05)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.figtext(0.25, 0.90,f'{pos} vs {neg}',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=10)
    plt.savefig(f'output/{name}{pos}{neg}_ROC.png')
    plt.close()

def doRocs(data, name, folds):
    rocGraph(data, (3, 2), name, folds)
    rocGraph(data, (3, 1), name, folds)
    rocGraph(data, (2, 3), name, folds)
    rocGraph(data, (2, 1), name, folds)
    rocGraph(data, (1, 2), name, folds)
    rocGraph(data, (1, 3), name, folds)



def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--name', default="")
    args.add_argument('--folds', default=10)
    return args.parse_args()

def _main():

    splitdata = {'IBL':[], 'Barrel':[], 'Endcap':[]}
    args = _get_args()

    for i in range(args.folds):

        with h5.File(f'output/{args.input}{i+1}.h5', 'r') as data:
            data_EC = data['NN_barrelEC'][()]
            data_Layer = data['NN_layer'][()]
            data_number = data['Output_number'][()]
            data_true = data['Output_number_true'][()]

        data = doNumber(data_EC, data_Layer, data_number, data_true)
        for j, layer in enumerate(data):
            splitdata[layer].append(data[layer])
    doRocs(splitdata, args.name, args.folds)

if __name__ == '__main__':
    _main()