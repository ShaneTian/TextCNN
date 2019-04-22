# TextCNN
TextCNN by TensorFlow 2.0.0 ( tf.keras mainly ).
## Software environments
1. tensorflow-gpu 2.0.0-alpha0
2. python 3.6.7
3. pandas 0.24.2
4. numpy 1.16.2

## Data
- Vocabulary size: 3447
- Number of classes: 18
- Train/Test split: 22386/226
- Shape of train data: (22386, 128)
- Shape of test data: (226, 128)

## Model architecture
![Model architecture](https://ws1.sinaimg.cn/large/006tNc79gy1g2bu549arhj316i0u0wmt.jpg)

## Run
### Train result
Use 22386 samples after 25 epochs:

| Loss | Accuracy | Precision | Recall | Val loss | Val accuracy | Val precision | Val recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1229 | 0.9790 | 0.9825 | 0.9743 | 0.2873 | 0.9286 | 0.9367 | 0.9241 |
### Test result
Use 226 samples:

| Accuracy | Precision | Recall | F1-Measure |
| --- | --- | --- | --- |
| 0.9381 | 0.9591 | 0.9336 | **0.9462** |
### images
**Loss**

![Loss](https://ws4.sinaimg.cn/large/006tNc79gy1g2buxirtkej30f709zdge.jpg)

**Accuracy**

![Accuracy](https://ws3.sinaimg.cn/large/006tNc79gy1g2buxixtiaj30ff09zaaw.jpg)

**Precision**

![Precision](https://ws2.sinaimg.cn/large/006tNc79gy1g2buxilmstj30fh09wjs5.jpg)

**Recall**

![Recall](https://ws4.sinaimg.cn/large/006tNc79gy1g2buxj23njj30fh0a0q3m.jpg)
### Usage
```
usage: train.py [-h] [-t TEST_SAMPLE_PERCENTAGE] [-p PADDING_SIZE]
                [-e EMBED_SIZE] [-f FILTER_SIZES] [-n NUM_FILTERS]
                [-d DROPOUT_RATE] [-c NUM_CLASSES] [-l REGULARIZERS_LAMBDA]
                [-b BATCH_SIZE] [--epochs EPOCHS]
                [--fraction_validation FRACTION_VALIDATION]
                [--log_dir LOG_DIR]

This is the TextCNN train project.

optional arguments:
  -h, --help            show this help message and exit
  -t TEST_SAMPLE_PERCENTAGE, --test_sample_percentage TEST_SAMPLE_PERCENTAGE
                        The fraction of test data.(default=0.01)
  -p PADDING_SIZE, --padding_size PADDING_SIZE
                        Padding size of sentences.(default=128)
  -e EMBED_SIZE, --embed_size EMBED_SIZE
                        Word embedding size.(default=128)
  -f FILTER_SIZES, --filter_sizes FILTER_SIZES
                        Convolution kernel sizes.(default=[3, 4, 5])
  -n NUM_FILTERS, --num_filters NUM_FILTERS
                        Number of each convolution kernel.(default=128)
  -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                        Dropout rate in softmax layer.(default=0.5)
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of target classes.(default=18)
  -l REGULARIZERS_LAMBDA, --regularizers_lambda REGULARIZERS_LAMBDA
                        L2 regulation parameter.(default=0)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Mini-Batch size.(default=64)
  --epochs EPOCHS       Number of epochs.(default=200)
  --fraction_validation FRACTION_VALIDATION
                        The fraction of validation.(default=0.01)
  --log_dir LOG_DIR     Log dir for tensorboard.(default=./log/)
```

```
usage: test.py [-h] [-p PADDING_SIZE] [-c NUM_CLASSES]

This is the TextCNN test project.

optional arguments:
  -h, --help            show this help message and exit
  -p PADDING_SIZE, --padding_size PADDING_SIZE
                        Padding size of sentences.(default=128)
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of target classes.(default=18)
```
#### You need to know...
1. You need to alter `load_data_and_write_to_file` function in `data_helper.py` to match you data file;
2. This code used single channel input, you can use two channels from embedding vector, one is static and the other is dynamic. Maybe it is greater;
3. The model is saved by `hdf5` file;
4. Tensorboard is available.
