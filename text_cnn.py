import argparse
import os
import re
import pandas as pd
import csv
from tensorflow import keras
import tensorflow as tf
import numpy as np


def text_preprocess(text):
    """
    Clean and segment the text.
    Return a new text.
    """
    text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+——！:；，。？、~@#%……&*（）·¥\-\|\\《》〈〉～]", "", text)
    text = re.sub("[<>]", "", text)
    text = re.sub("[a-zA-Z0-9]", "", text)
    text = re.sub(r"\s", "", text)
    if not text:
        return ''
    return ' '.join(string for string in text)


def load_data_and_write_to_file(data_file, out_file):
    """
    Loads xlsx from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load and clean data from files
    case_type = ['民事案件', '刑事案件', '行政案件', '赔偿案件', '执行案件']
    df = pd.read_excel(data_file, sheet_name=case_type, usecols=[3, 5], dtype=str)
    x_text, y = [], []
    for each_case_type in case_type:
        x_text += df[each_case_type]["自然段正文"].to_list()
        y += df[each_case_type]["正确分段标记"].to_list()
    x_new = []
    empty_idx = []
    for idx, each_text in enumerate(x_text):
        tmp = text_preprocess(each_text)
        if tmp:
            x_new.append(tmp)
        else:
            empty_idx.append(idx)

    # Generate labels
    y_new = []
    for idx, label in enumerate(y):
        if idx in empty_idx:
            continue
        label = label.split('，')[0]
        if label == '99':
            y_new.append(0)
        else:
            y_new.append(int(label))

    # Write to CSV file
    with open(out_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(zip(x_new, y_new))
    return [x_new, y_new]


def preprocess(data_file):
    """
    Text to sequence, compute max document length and vocabulary size,
    padding for x by max document length, and split data to train/test.
    Returns: x_train, y_train, x_test, y_test, vocab_size, max_doc_length
    """
    print("Loading data...")
    df = pd.read_csv(data_file, header=None, names=["x_text", "y_label"])
    x_text, y = df["x_text"].to_list(), df["y_label"].to_list()

    # Texts to sequences
    text_preprocesser = keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
    text_preprocesser.fit_on_texts(x_text)
    x = text_preprocesser.texts_to_sequences(x_text)
    word_dict = text_preprocesser.word_index
    vocab_size = len(word_dict)
    max_doc_length = max([len(each_text) for each_text in x])

    # Padding sequences by max_doc_length
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_doc_length,
                                                   padding='post', truncating='post')

    # Shuffle and split data
    np.random.seed(323)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = np.array(x)[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]
    test_sample_index = -1 * int(args.test_sample_percentage * len(y))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
    del x, y, x_shuffled, y_shuffled

    print("Vocabulary size: {:d}".format(vocab_size))
    print("Max document length: {:d}".format(max_doc_length))
    print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    print("Shape of train data: {}".format(np.shape(x_train)))
    print("Shape of test data: {}\n".format(np.shape(x_test)))
    return x_train, y_train, x_test, y_test, vocab_size, max_doc_length


def TextCNN(vocab_size, feature_size):
    inputs = keras.Input(shape=(feature_size,))
    embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
    embed = keras.layers.Embedding(vocab_size, args.embed_size,
                                   embeddings_initializer=embed_initer, input_length=feature_size)(inputs)
    # single channel. If using real embedding, you can set one static
    embed = keras.layers.Reshape((feature_size, args.embed_size, 1))(embed)

    pool_outputs = []
    for filter_size in args.filter_sizes:
        filter_shape = (filter_size, args.embed_size)
        conv = keras.layers.Conv2D(args.num_filters, filter_shape, strides=(1, 1), padding='valid',
                                   data_format='channels_last', activation='relu',
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1))(embed)
        max_pool_shape = (feature_size - filter_size + 1, 1)
        pool = keras.layers.MaxPool2D(pool_size=max_pool_shape, strides=(1, 1), padding='valid',
                                      data_format='channels_last')(conv)
        pool_outputs.append(pool)

    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1)
    pool_outputs = keras.layers.Flatten(data_format='channels_last')(pool_outputs)
    pool_outputs = keras.layers.Dropout(args.dropout_rate)(pool_outputs)

    outputs = keras.layers.Dense(args.num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(args.regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(args.regularizers_lambda))(pool_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(x_train, y_train, vocab_size, feature_size, save_path):
    print("\nTrain...")
    model = TextCNN(vocab_size, feature_size)
    model.summary()
    parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    parallel_model.compile(tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    # keras.utils.plot_model(model, show_shapes=True, to_file="../model.png")
    y_train = tf.one_hot(y_train, args.num_classes)
    tb_callback = keras.callbacks.TensorBoard(args.log_dir, histogram_freq=0.1, write_graph=True,
                                              write_grads=True, write_images=True, embeddings_freq=0.5, update_freq='batch')
    history = parallel_model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
                                 callbacks=[tb_callback], validation_split=args.fraction_validation, shuffle=True)
    print("\nSaving model...")
    keras.models.save_model(model, save_path)
    print(history.history)


def test(model, x_test, y_test):
    print("\nTesting...")
    y_pred = model.predict(x=x_test, batch_size=1, verbose=1)
    y_test = tf.one_hot(y_test, args.num_classes)
    m = keras.metrics.CategoricalAccuracy()
    m.update_state(y_test, y_pred)
    print("Test accuracy: {:f}".format(m.result()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a TextCNN.')
    parser.add_argument('-t', '--test_sample_percentage', default=0.01, type=float, help='The fraction of test data.(default=0.01)')
    parser.add_argument('-e', '--embed_size', default=128, type=int, help='Word embedding size.(default=128)')
    parser.add_argument('-f', '--filter_sizes', default=[3, 4, 5], type=list, help='Convolution kernel sizes.(default=[3, 4, 5])')
    parser.add_argument('-n', '--num_filters', default=128, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.5, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=18, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs.(default=200)')
    parser.add_argument('--fraction_validation', default=0.01, type=float, help='The fraction of validation.(default=0.01)')
    parser.add_argument('--log_dir', default='./log/', type=str, help='Log dir for tensorboard.(default=./log/)')
    # parser.add_argument('--test', action='store_true', help='Whether to test the model.(default=False)')
    args = parser.parse_args()

    if not os.path.exists("./data.csv"):
        load_data_and_write_to_file("../fenduan_clean.xlsx", "./data.csv")
    x_train, y_train, x_test, y_test, vocab_size, max_doc_length = preprocess("./data.csv")

    train(x_train, y_train, vocab_size, max_doc_length, "./results/TextCNN.h5")
    print("\nLoading model...")
    model = keras.models.load_model("./results/TextCNN.h5")
    test(model, x_test, y_test)
