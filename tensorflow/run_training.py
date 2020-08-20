import numpy as np
import h5py as h5
import logging
import datetime
from collections import namedtuple
import argparse
#import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
#from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import GlorotNormal
from tensorflow import random_normal_initializer


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--inputData', required=True)
    args.add_argument('--outputModel', required=True)
    args.add_argument('--structure', default=[25, 20])
    args.add_argument('--learning_rate', default=0.08)
    args.add_argument('--regularizer', default=1e-7)
    args.add_argument('--batch_size', default=60)
    args.add_argument('--epochs', default=50)
    args.add_argument('--momentum', default=0.4)
    args.add_argument('--patience', default=10)
    args.add_argument('--hidden_activation', default=_sigmoid2) #tf.keras.activations.sigmoid,
    args.add_argument('--output_activation', default=_sigmoid2) #tf.keras.activations.sigmoid,
    return args.parse_args()

def _main():
    args = _get_args()

    '''
    Option = namedtuple("MyStruct", "data_input model_output structure learning_rate regularizer batch_size epochs \
                    momentum patience hidden_activation output_activation")
                    # folds batch_size epochs")
                    #structure learning_rate regularizer epochs")
    args = Option(
        data_input='C:/Users/alexc/Desktop/Projects/pixel-NN-master/data/814_evensplit_equalfrac_train.h5',
        model_output='output/tf_model',
        structure=[25, 20],
        learning_rate=0.08,
        regularizer=1e-7,
        batch_size=60,
        epochs=20,
        momentum=0.4,
        patience=50,
        hidden_activation=_sigmoid2, #tf.keras.activations.sigmoid,
        output_activation=_sigmoid2 #tf.keras.activations.sigmoid,
    )
    '''

    logging.basicConfig(
        level='INFO',
        format='[%(asctime)s %(levelname)s] %(message)s'
    )

    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    logging.info(f'Loading data from {args.inputData}')

    with h5.File(args.inputData, 'r') as data:
        data_x = data['input'][()]
        data_y = data['target'][()]
        #data_x = data['input'][:10000]
        #data_y = data['target'][:10000]

    std = np.std(data_x, axis=0, ddof=1)
    std[np.where(std == 0)] = 1
    init = random_normal_initializer(seed=42)

    inputs = Input((data_x.shape[1],))

    h = OffsetAndScale(offset=-np.mean(data_x, axis=0), scale=1.0/std)(inputs)

    for l in range(0, len(args.structure)):
        h = Dense(args.structure[l], activation=args.hidden_activation, kernel_regularizer=l2(args.regularizer), kernel_initializer=init)(h)
        #h = args.hidden_activation(h) #, alpha=0.1) #for leaky relu
        
    output_layer = Dense(data_y.shape[1], activation=args.output_activation, kernel_regularizer=l2(args.regularizer), kernel_initializer=init)(h)
    #output_layer = args.output_activation(output_layer)

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)

    print(model.summary())
    keras.utils.plot_model(model)

    compile_args = {
        'optimizer': SGD(learning_rate=args.learning_rate, momentum=args.momentum),
        'loss': 'categorical_crossentropy'
    }

    model.compile(**compile_args)

    #log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    fit_args = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'callbacks': [
            EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, mode='auto'),
            ModelCheckpoint(args.outputModel+'.h5', save_best_only=True, verbose=2),
            #TensorBoard(log_dir=log_dir, histogram_freq=1)
        ],
        'verbose': 1,
        'validation_split': 0.1 #not compatible with fit_args when using tensordata
    }

    history = model.fit(data_x, data_y, **fit_args)

    #fit_args.pop('batch_size'), fit_args.pop('validation_split')
    #validation_index = int(data_x.shape[0]*0.1)
    #validationDataSet = tf.data.Dataset.from_tensor_slices((data_x[:validation_index], data_y[:validation_index])).batch(args.batch_size)
    #trainDataSet = tf.data.Dataset.from_tensor_slices((data_x[validation_index:], data_y[validation_index:])).batch(args.batch_size)
    #history = model.fit(x=trainDataSet, validation_data=validationDataSet, **fit_args)
    
    hpath = args.outputModel + '.history.h5'
    logging.info('Writing fit history to %s', hpath)
    with h5.File(hpath, 'w') as hfile:
        for key, val in history.history.items():
            hfile.create_dataset(key, data=np.array(val))


def _sigmoid2(x):
    return tf.math.sigmoid(2*x)

def _config(layer, config):
    base_config = super(layer.__class__, layer).get_config()
    conf_dict = dict(base_config.items()).copy()
    conf_dict.update(dict(config.items()))
    return conf_dict

class OffsetAndScale(keras.layers.Layer):
    """ (x + offset) * scale """

    def __init__(self, offset, scale, **kwargs):
        self.offset = offset
        self.scale = scale

        if isinstance(self.scale, dict) and self.scale['type'] == 'ndarray':
            self.scale = np.array(self.scale['value']).astype('float32')

        if isinstance(self.offset, dict) and self.offset['type'] == 'ndarray':
            self.offset = np.array(self.offset['value']).astype('float32')

        super(OffsetAndScale, self).__init__(**kwargs)

    def call(self, x):
        return (x + self.offset) * self.scale

    def get_config(self):
        return _config(self, {
            'offset': self.offset,
            'scale': self.scale
        })

'''
#FOR APPLYING MDOEL AND PLOTTING ROCS

def _do_number(data_x, data_y, pred, thrs):

    outdata = np.zeros(
        (pred.shape[0],),
        dtype=[
            ('Output_number', np.dtype(('f4', 3))),
            ('Output_number_true', 'i4'),
            ('Output_number_estimated', 'i4'),
            ('NN_layer', 'i4'),
            ('NN_barrelEC', 'i4'),
            ('globalEta', 'f4'),
            ('globalPhi', 'f4'),
            ('cluster_size', 'i4'),
            ('cluster_size_X', 'i4'),
            ('cluster_size_Y', 'i4'),
        ]
    )

    outdata['Output_number'] = pred / np.sum(pred, axis=1, keepdims=True)
    outdata['Output_number_true'] = 1 * data_y[:,0] + \
                                    2 * data_y[:,1] + \
                                    3 * data_y[:,2]

    outdata['Output_number_estimated'][
        np.where(np.logical_and(pred[:,1] < thrs[0], pred[:,2] < thrs[1]))
    ] = 1
    outdata['Output_number_estimated'][
        np.where(np.logical_and(pred[:,1] >= thrs[0], pred[:,2] < thrs[1]))
    ] = 2
    outdata['Output_number_estimated'][
        np.where(pred[:,2] >= thrs[1])
    ] = 3

    auxfields = [
        'NN_layer',
        'NN_barrelEC',
        'globalEta',
        'globalPhi',
        'cluster_size',
        'cluster_size_X',
        'cluster_size_Y'
    ]

    outdata['NN_layer'] = data_x[:, 56]
    outdata['NN_barrelEC'] = data_x[:, 57]
    #for field in auxfields:
    #    outdata[field] = data_x[field]

    return outdata

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

def rocGraph(data, classes, name):

    fpr = [[0]*(10) for i in range(len(data))]
    tpr = [[0]*(10) for i in range(len(data))]
    auc1 = [[0]*(10) for i in range(len(data))]

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

        for j in range(10):
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
    
def doRocs(data, name):
    rocGraph(data, (3, 2), name)
    rocGraph(data, (3, 1), name)
    rocGraph(data, (2, 3), name)
    rocGraph(data, (2, 1), name)
    rocGraph(data, (1, 2), name)
    rocGraph(data, (1, 3), name)

'''

if __name__ == "__main__":
    _main()