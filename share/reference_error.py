import PixelNN.keras_utils as keras_utils

def build_model(data_x, data_y):
    return keras_utils.simple_model(
        data_x,
        data_y,
        structure=[15, 10],
        hidden_activation=keras_utils.Sigmoid2,
        output_activation=keras_utils.Sigmoid2,
        learning_rate=0.3,
        weight_decay=1e-6,
        momentum=0.7,
        minibatch_size=50,
        loss_function='categorical_crossentropy'
    )
