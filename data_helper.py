import re
import pandas as pd
import csv
from tensorflow.keras import preprocessing
import numpy as np
import json


def text_preprocess(text):
    """
    Clean and segment the text.
    Return a new text.
    """
    text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+——！:；，。？、~@#%……&*（）·¥\-\|\\《》〈〉～]",
                  "", text)
    text = re.sub("[<>]", "", text)
    text = re.sub("[a-zA-Z0-9]", "", text)
    text = re.sub(r"\s", "", text)
    if not text:
        return ''
    return ' '.join(string for string in text)


def load_data_and_write_to_file(data_file, train_data_file, test_data_file, test_sample_percentage):
    """
    Loads xlsx from files, splits the data to train and test data, write them to file.
    """
    # Load and clean data from files
    case_type = ['民事案件', '刑事案件', '行政案件', '赔偿案件', '执行案件']
    df = pd.read_excel(data_file, sheet_name=case_type, usecols=[3, 5], dtype=str)
    x_text, y = [], []
    for each_case_type in case_type:
        x_text += df[each_case_type]["自然段正文"].tolist()
        y += df[each_case_type]["正确分段标记"].tolist()
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

    # Shuffle data and split data to train and test
    np.random.seed(323)
    np.random.shuffle(x_new)
    np.random.seed(323)
    np.random.shuffle(y_new)
    test_sample_index = -1 * int(test_sample_percentage * len(y_new))
    x_train, x_test = x_new[:test_sample_index], x_new[test_sample_index:]
    y_train, y_test = y_new[:test_sample_index], y_new[test_sample_index:]

    # Write to CSV file
    with open(train_data_file, 'w', newline='', encoding='utf-8-sig') as f:
        print('Write train data to {} ...'.format(train_data_file))
        writer = csv.writer(f)
        writer.writerows(zip(x_train, y_train))
    with open(test_data_file, 'w', newline='', encoding='utf-8-sig') as f:
        print('Write test data to {} ...'.format(test_data_file))
        writer = csv.writer(f)
        writer.writerows(zip(x_test, y_test))


def preprocess(data_file, vocab_file, padding_size, test=False):
    """
    Text to sequence, compute vocabulary size, padding sequence.
    Return sequence and label.
    """
    print("Loading data from {} ...".format(data_file))
    df = pd.read_csv(data_file, header=None, names=["x_text", "y_label"])
    x_text, y = df["x_text"].tolist(), df["y_label"].tolist()

    if not test:
        # Texts to sequences
        text_preprocesser = preprocessing.text.Tokenizer(oov_token="<UNK>")
        text_preprocesser.fit_on_texts(x_text)
        x = text_preprocesser.texts_to_sequences(x_text)
        word_dict = text_preprocesser.word_index
        json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
        vocab_size = len(word_dict)
        # max_doc_length = max([len(each_text) for each_text in x])
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        print("Vocabulary size: {:d}".format(vocab_size))
        print("Shape of train data: {}".format(np.shape(x)))
        return x, y, vocab_size
    else:
        word_dict = json.load(open(vocab_file, 'r'))
        vocabulary = word_dict.keys()
        x = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for each_sentence in x_text]
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        print("Shape of test data: {}\n".format(np.shape(x)))
        return x, y
