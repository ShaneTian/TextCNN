from tensorflow import keras


def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,
            filter_sizes, regularizers_lambda, dropout_rate):
    inputs = keras.Input(shape=(feature_size,), name='input_data')
    embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
    embed = keras.layers.Embedding(vocab_size, embed_size,
                                   embeddings_initializer=embed_initer,
                                   input_length=feature_size,
                                   name='embedding')(inputs)
    # single channel. If using real embedding, you can set one static
    embed = keras.layers.Reshape((feature_size, embed_size, 1), name='add_channel')(embed)

    pool_outputs = []
    for filter_size in list(map(int, filter_sizes.split(','))):
        filter_shape = (filter_size, embed_size)
        conv = keras.layers.Conv2D(num_filters, filter_shape, strides=(1, 1), padding='valid',
                                   data_format='channels_last', activation='relu',
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1),
                                   name='convolution_{:d}'.format(filter_size))(embed)
        max_pool_shape = (feature_size - filter_size + 1, 1)
        pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                      strides=(1, 1), padding='valid',
                                      data_format='channels_last',
                                      name='max_pooling_{:d}'.format(filter_size))(conv)
        pool_outputs.append(pool)

    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
    pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
    pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)

    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(pool_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
