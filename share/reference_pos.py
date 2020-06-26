import PixelNN.keras_utils as keras_utils

def build_model(data_x, data_y):
    return keras_utils.simple_model(
        data_x,
        data_y,
        structure=[40, 20],
        hidden_activation=keras_utils.Sigmoid2,
        output_activation=None,
        learning_rate=0.04,
        weight_decay=1e-7,
        momentum=0.3,
        minibatch_size=30,
        loss_function='mean_squared_error'
    )
