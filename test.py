import argparse
from data_helper import preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import tensorflow as tf


def test(model, x_test, y_test):
    print("Test...")
    y_pred = model.predict(x=x_test, batch_size=1, verbose=1)
    y_test = tf.one_hot(y_test, args.num_classes)
    m1 = metrics.CategoricalAccuracy()
    m2 = metrics.Precision()
    m3 = metrics.Recall()
    m1.update_state(y_test, y_pred)
    m2.update_state(y_test, y_pred)
    m3.update_state(y_test, y_pred)
    print("Test accuracy: {:f}".format(m1.result()))
    print("Test precision: {:f}".format(m2.result()))
    print("Test recall: {:f}".format(m3.result()))
    print("Test F1-Measure: {:f}".format(2 * m2.result() * m3.result() / (m2.result() + m3.result())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('-p', '--padding_size', default=128, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-c', '--num_classes', default=18, type=int, help='Number of target classes.(default=18)')
    args = parser.parse_args()

    x_test, y_test = preprocess("./data/test_data.csv", "./results/vocab.json", args.padding_size, test=True)
    print("Loading model...")
    model = load_model("./results/TextCNN.h5")
    test(model, x_test, y_test)
