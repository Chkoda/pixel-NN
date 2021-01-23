import sys
sys.path.append('scripts')
sys.path.append('share')
sys.path.append('python')
import os
import argparse
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def doNumber(data_EC, data_Layer, data_number, data_true, data_estimated):

    data = {}

    outdata = np.zeros(
        (data_number.shape[0],),
        dtype=[
            ('Output_number', np.dtype(('f4', 3))),
            ('Output_number_true', 'i4'),
            ('Output_number_estimated', 'i4')
        ]
    )

    IBL = np.logical_and(data_Layer == 0, data_EC == 0)
    Barrel = np.logical_and(data_Layer > 0, data_EC == 0)
    Endcap = data_EC != 0

    outdata['Output_number'] = data_number
    outdata['Output_number_true'] = data_true
    outdata['Output_number_estimated'] = data_estimated

    '''
    thrs = [0.6, 0.2]
    outdata['Output_number_estimated'][
        np.where(np.logical_and(data_number[:,1] < thrs[0], data_number[:,2] < thrs[1]))
    ] = 1
    outdata['Output_number_estimated'][
        np.where(np.logical_and(data_number[:,1] >= thrs[0], data_number[:,2] < thrs[1]))
    ] = 2
    outdata['Output_number_estimated'][
        np.where(data_number[:,2] >= thrs[1])
    ] = 3
    '''

    data['IBL'] = outdata[IBL]
    data['Barrel'] = outdata[Barrel]
    data['Endcap'] = outdata[Endcap]
    return data

def evaluate(data, classes):
    pos, neg = classes

    pos_sel = data['Output_number_true'] == pos
    neg_sel = data['Output_number_true'] == neg

    isel = np.where(
        np.logical_or(
            pos_sel,
            neg_sel,
        )
    )[0]

    fpr, tpr, _ = roc_curve(
        data['Output_number_true'][isel],
        data['Output_number'][isel][:, pos - 1],
        pos_label=pos
    )
    auc1 = auc(fpr, tpr)

    return fpr, tpr, auc1

def plotRocs(fpr, tpr, auc1, labels, classes, outpath, title):
    pos, neg = classes
    plt.style.use('classic')

    for i, label in enumerate(labels):
        plt.plot(fpr[i], tpr[i], label='{} (AUC = {:.2f})'.format(label, auc1[i]))
    
    rand_chance = np.linspace(0, 1, 100)
    plt.plot(rand_chance, rand_chance, ':', color='grey', label='Random (AUC = 0.5)')
    plt.semilogx()
    plt.ylabel("Pr(Estimated: {}-particle | True: {}-particle)".format(pos, pos))
    plt.xlabel("Pr(Estimated: {}-particle | True: {}-particle)".format(pos, neg))
    plt.xlim([0.001, 1.05])
    plt.ylim(0,1.5)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.figtext(0.25, 0.90,'{} {} vs {}'.format(title, pos, neg),fontweight='bold', wrap=True, horizontalalignment='right', fontsize=10)
    plt.savefig('{}{}_{}vs{}_ROC.png'.format(outpath, title, pos, neg))
    plt.close()

def rocGraph(datas, labels, outpath, classes):

    fpr = [None] * len(datas)
    tpr = [None] * len(datas)
    auc1 = [None] * len(datas)

    fprl = []
    tprl = []
    aucl = []

    for j, layer in enumerate(datas[0]):
        for i, data in enumerate(datas):
            fpr[i], tpr[i], auc1[i] = evaluate(data[layer], classes)
        fprl.append(fpr)
        tprl.append(tpr)
        aucl.append(auc1)
        
    for i, layer in enumerate(datas[0]):
        plotRocs(fprl[i], tprl[i], aucl[i], labels, classes, outpath, layer)


    for i, data in enumerate(datas):
        fpr[i], tpr[i], auc1[i] = evaluate(np.concatenate([data[x] for x in data]), classes)

    plotRocs(fpr, tpr, auc1, labels, classes, outpath, 'All')
        
def doRocs(datas, labels, outpath):
    rocGraph(datas, labels, outpath, (3, 2))
    rocGraph(datas, labels, outpath, (3, 1))
    rocGraph(datas, labels, outpath, (2, 3))
    rocGraph(datas, labels, outpath, (2, 1))
    rocGraph(datas, labels, outpath, (1, 2))
    rocGraph(datas, labels, outpath, (1, 3))


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', default='output/')
    args.add_argument('--output')

    grp = args.add_mutually_exclusive_group(required=True)
    grp.add_argument('--model')
    grp.add_argument('--models', nargs='+')

    grp2 = args.add_mutually_exclusive_group(required=False)
    grp2.add_argument('--label')
    grp2.add_argument('--labels', nargs='+')

    return args.parse_args()

def _main():

    args = _get_args()
    
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models

    if args.label:
        labels = [args.label]
    elif args.labels:
        labels = args.labels
    else:
        labels = models
    
    datas = []

    for model in models:
        with h5.File('{}{}'.format(args.input, model), 'r') as data:
            data_EC = data['NN_barrelEC'][()]
            data_Layer = data['NN_layer'][()]
            data_number = data['Output_number'][()]
            data_true = data['Output_number_true'][()]
            data_estimated = data['Output_number_estimated'][()]

        datas.append(doNumber(data_EC, data_Layer, data_number, data_true, data_estimated))

    if args.output:
        outpath = args.output
    else:
        outpath = ''
        for i in args.input.split('/')[:-1]:
            outpath = outpath + i + '/'
        outpath += 'rocs/'

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    logging.info('Output path: %s', outpath)

    print('Created ROC plots in {}'.format(outpath))
    doRocs(datas, labels, outpath)

if __name__ == '__main__':
    _main()
