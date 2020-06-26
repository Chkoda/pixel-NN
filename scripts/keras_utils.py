import logging
import warnings

import keras
import keras.backend as K
import numpy as np


def load_model(path):
    return keras.models.load_model(
        path,
        custom_objects={
            'OffsetAndScale': OffsetAndScale,
            '_sigmoid2': _sigmoid2
        }
    )


def simple_model(data_x,
                 data_y,
                 structure,
                 hidden_activation,
                 output_activation,
                 learning_rate,
                 weight_decay,
                 momentum,
                 minibatch_size,
                 loss_function):

    input_node = keras.layers.Input((data_x.shape[1],))

    std = np.std(data_x, axis=0, ddof=1)
    std[np.where(std == 0)] = 1
    model = OffsetAndScale(
        offset=-np.mean(data_x, axis=0),
        scale=1.0/std
    )(input_node)

    for n in structure:
        model = keras.layers.Dense(
            units=n,
            kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(model)
        model = hidden_activation(model)

    model = keras.layers.Dense(
        units=data_y.shape[1],
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )(model)

    if output_activation:
        model = output_activation(model)

    model = keras.models.Model(inputs=input_node, outputs=model)

    compile_args = {
        'optimizer': keras.optimizers.SGD(
            lr=learning_rate,
            momentum=momentum
        ),
        'loss': loss_function
    }

    fit_args = {
        'batch_size': minibatch_size,
        'epochs': 1000,
        'callbacks': [
            ThresholdEarlyStopping(verbose=1, min_epochs=50),
        ],
        'validation_split': 0.1,
    }

    return model, compile_args, fit_args, None


def _sigmoid2(x):
    import sys
    MAXEXP = np.log(sys.float_info.max)
    return K.switch(
        K.greater_equal(-2*x, MAXEXP),
        0.0 * x,
        1.0 / (1.0 + K.exp(-2*x))
    )


Sigmoid2 = keras.layers.Activation(_sigmoid2)


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


class ThresholdEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_epochs=10,
                 threshold=0.995, increase=1.75, verbose=0, mode='auto'):

        super(ThresholdEarlyStopping, self).__init__(
            monitor=monitor,
            patience=min_epochs,
            verbose=verbose,
            mode=mode
        )

        self.threshold = threshold
        self.increase = increase

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.patience:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Early stopping requires %s available!' %
                              (self.monitor), RuntimeWarning)

            if self.monitor_op(current, self.best):
                if self.monitor_op(current, self.threshold*self.best):
                    self.patience = max(self.patience, epoch * self.increase)
                self.best = current

        else:
            if self.verbose > 0:
                print('Epoch %05d: early stopping' % (epoch))

            self.model.stop_training = True
