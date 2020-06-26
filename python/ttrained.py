import collections

import keras
import numpy as np
import ROOT

from keras_utils import OffsetAndScale, Sigmoid2

# Force ROOT to load the TTrainedNetwork code
ROOT.TTrainedNetwork()

def to_keras(ttnet):
    ninput = ttnet.getnInput()
    nhidden = list(ttnet.getnHiddenLayerSize())
    noutput = ttnet.getnOutput()

    input_node = keras.models.Input((ninput,))

    norm = ROOT.TTrainedNetworkNormalization(ttnet)
    offsets = np.array(norm.offsets())
    scales = np.array(norm.scales())

    net = OffsetAndScale(offsets, scales)(input_node)

    for n in nhidden:
        net = keras.layers.Dense(n)(net)
        net = Sigmoid2(net)
    net = keras.layers.Dense(noutput)(net)
    if not ttnet.getIfLinearOutput():
        net = Sigmoid2(net)

    model = keras.models.Model(inputs=input_node, outputs=net)

    weights = []
    for mat, vec in zip(ttnet.weightMatrices(), ttnet.getThresholdVectors()):
        weights.append(np.empty((mat.GetNrows(), mat.GetNcols())))
        for i in range(mat.GetNrows()):
            for j in range(mat.GetNcols()):
                weights[-1][i,j] = mat(i,j)
        weights.append(np.empty(vec.GetNrows()))
        for i in range(vec.GetNrows()):
            weights[-1][i] = vec(i)

    model.set_weights(weights)


    return model


def from_keras(knet):

    cfg = knet.get_config()
    layers = cfg['layers']

    nInput = knet.get_input_shape_at(0)[-1]
    nHidden = len([l for l in layers if l['class_name'] == 'Dense']) - 1
    nOutput = layers[-1]['config']['units']
    nHiddenLayerSize = _get_nHiddenLayerSize(knet)
    weightMatrices, thresholdVectors = _get_weights_biases(knet)
    activationFunction = 1  # Sigmoid2
    linearOutput = not any(['activation' in s[0] for s in cfg['output_layers']])
    normalizeOutput = False
    offsets, scales = _get_offsets_scales(knet)

    ttnet = ROOT.TTrainedNetwork(
        nInput,
        nHidden,
        nOutput,
        nHiddenLayerSize,
        thresholdVectors,
        weightMatrices,
        activationFunction,
        linearOutput,
        normalizeOutput
    )
    ttnet.setOffsets(offsets)
    ttnet.setScales(scales)

    return ttnet


def _get_nHiddenLayerSize(knet):
    vec = ROOT.vector('Int_t')()
    units = [
        l['config']['units'] for l in knet.get_config()['layers']
        if l['class_name'] == 'Dense'
    ]
    for u in units[:-1]:
        vec.push_back(u)
    return vec

def _get_weights_biases(knet):
    all_weights = ROOT.vector('TMatrixT<double>*,allocator<TMatrixT<double>*> ')()
    all_biases = ROOT.vector('TVectorT<double>*,allocator<TVectorT<double>*> ')()
    weights_biases = knet.get_weights()

    for i in range(0, len(weights_biases), 2):
        weights = weights_biases[i]
        matrix = ROOT.TMatrixD(*weights.shape)
        for m in range(weights.shape[0]):
            for n in range(weights.shape[1]):
                matrix[m][n] = weights[m, n]
        all_weights.push_back(matrix)
    for i in range(1, len(weights_biases), 2):
        biases = weights_biases[i]
        vector = ROOT.TVectorD(biases.shape[0])
        for m in range(biases.shape[0]):
            vector[m] = biases[m]
        all_biases.push_back(vector)
    return all_weights, all_biases


def _get_offsets_scales(knet):

    for lyr in knet.get_config()['layers']:
        if lyr['class_name'] == 'OffsetAndScale':
            break

    offsets = ROOT.vector('double')()
    for v in lyr['config']['offset']:
        offsets.push_back(v)

    scales = ROOT.vector('double')()
    for v in lyr['config']['scale']:
        scales.push_back(v)

    return offsets, scales


nn_key_dict = collections.OrderedDict([
    ('number', 'NumberParticles'),
    ('pos1', 'ImpactPoints1P'),
    ('pos2', 'ImpactPoints2P'),
    ('pos3', 'ImpactPoints3P'),
    ('error1x', 'ImpactPointErrorsX1'),
    ('error2x', 'ImpactPointErrorsX2'),
    ('error3x', 'ImpactPointErrorsX3'),
    ('error1y', 'ImpactPointErrorsY1'),
    ('error2y', 'ImpactPointErrorsY2'),
    ('error3y', 'ImpactPointErrorsY3'),
])
