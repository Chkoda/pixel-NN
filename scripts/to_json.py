import glob
import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import h5py as h5
import imp
import keras
import keras.backend as K
DIR = "C:/Users/alexc/Desktop/Projects/pixel-NN-master/modelWeights/"


def _sigmoid2(x):
    import sys
    MAXEXP = np.log(sys.float_info.max)
    return K.switch(
        K.greater_equal(-2*x, MAXEXP),
        0.0 * x,
        1.0 / (1.0 + K.exp(-2*x))
    )

def _config(layer, config):
    base_config = super(layer.__class__, layer).get_config()
    return dict(base_config.items() + config.items())

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


def _build_model(path, data_x, data_y):
    """ Load a model from a .py source """
    #defm = SourceFileLoader('model_def', path).load_module()
    defm = imp.load_source('model_def', path)
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'" % path)
    buildf = defm.build_model
    return buildf(data_x, data_y)

def main():
    '''
    path=DIR+'kfold_evensplit/theano104_evensplit_kfold1.h5'
    kmodel = keras.models.load_model(
        path,
        #custom_objects={'OffsetAndScale': OffsetAndScale,
        #                '_sigmoid2': _sigmoid2
        #}
    )
    '''
    with h5.File('data/test.h5', 'r') as hfile:
        data_x = hfile['input'][:100]
        data_y = hfile['target'][:100]

    kmodel, compile_args, _, _ = _build_model('share/reference_number.py', data_x, data_y)
    kmodel.compile(**compile_args)
    
    outpath = 'output/LGconfigjson.json'
    json_config = kmodel.to_json()
    with open(outpath, 'w') as ofile:
        ofile.write(kmodel.to_json())


if __name__ == '__main__':
    main()
