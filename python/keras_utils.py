import logging
import warnings
import numpy as np
np.random.seed(42)

#import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.keras.backend as K

import keras
import keras.backend as K


def load_model(path):
    return keras.models.load_model(
        path
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
    
    model = keras.layers.Dense(
        units=25,
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(input_node)
    model = keras.layers.Activation(keras.activations.sigmoid)(model)

    model = keras.layers.Dense(
        units=20,
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(model)
    model = keras.layers.Activation(keras.activations.sigmoid)(model)


    model = keras.layers.Dense(
        units=data_y.shape[1],
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )(model)

    if output_activation:
        model = keras.layers.Activation(keras.activations.sigmoid)(model)

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
            ThresholdEarlyStopping(verbose=1, min_epochs=50)
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
#Sigmoid2 = keras.layers.Activation(keras.activations.sigmoid)


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

            elif self.monitor_op(current, self.best):
                if self.monitor_op(current, self.threshold*self.best):
                    self.patience = max(self.patience, epoch * self.increase)
                self.best = current

        else:
            if self.verbose > 0:
                print('Epoch %05d: early stopping' % (epoch))

            self.model.stop_training = True
