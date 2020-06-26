import PixelNN.keras_utils as keras_utils

def build_model(data_x, data_y):
    return keras_utils.simple_model(
        data_x,
        data_y,
        structure=[25, 20],
        hidden_activation=keras_utils.Sigmoid2,
        output_activation=keras_utils.Sigmoid2,
        learning_rate=0.08,
        weight_decay=1e-7,
        momentum=0.4,
        minibatch_size=60,
        loss_function='categorical_crossentropy'
    )
