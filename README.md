# TextCNN
TextCNN by TensorFlow 2.0.0 ( tf.keras mainly ).
## Software environments
1. tensorflow-gpu 2.0.0-alpha0
2. python 3.6.7
3. pandas 0.24.2
4. numpy 1.16.2

## Data
- Vocabulary size: 3407
- Number of classes: 18
- Train/Test split: 20351/2261

## Model architecture
![](https://ws1.sinaimg.cn/large/006tNc79gy1g2e66v8qatj311q0okao2.jpg)

## Model parameters
- Padding size: 128
- Embedding size: 512
- Num channel: 1
- Filter size: [3, 4, 5]
- Num filters: 128
- Dropout rate: 0.5
- Regularizers lambda: 0.01
- Batch size: 64
- Epochs: 10
- Fraction validation: 0.05 (1018 samples)
- Total parameters: 2,538,130

## Run
### Train result
Use 20351 samples after 10 epochs:

| Loss | Accuracy | Precision | Recall | Val loss | Val accuracy | Val precision | Val recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1078 | 0.9815 | 0.9844 | 0.9776 | 0.4020 | 0.9194 | 0.9292 | 0.9155 |
### Test result
Use 2261 samples:

| Accuracy | Precision | Recall | F1-Measure |
| --- | --- | --- | --- |
| 0.9371 | 0.9459 | 0.9279 | **0.9368** |
### Images
![](https://ws2.sinaimg.cn/large/006tNc79gy1g2e65013p0j30ew0k2dhp.jpg)

### Usage
```
usage: train.py [-h] [-t TEST_SAMPLE_PERCENTAGE] [-p PADDING_SIZE]
                [-e EMBED_SIZE] [-f FILTER_SIZES] [-n NUM_FILTERS]
                [-d DROPOUT_RATE] [-c NUM_CLASSES] [-l REGULARIZERS_LAMBDA]
                [-b BATCH_SIZE] [--epochs EPOCHS]
                [--fraction_validation FRACTION_VALIDATION]
                [--results_dir RESULTS_DIR]

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
                        Convolution kernel sizes.(default=3,4,5)
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
  --results_dir RESULTS_DIR
                        The results dir including log, model, vocabulary and
                        some inages.(default=./results/)
```

```
usage: test.py [-h] [-p PADDING_SIZE] [-c NUM_CLASSES] results_dir

This is the TextCNN test project.

positional arguments:
  results_dir           The results dir including log, model, vocabulary and
                        some inages.

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
