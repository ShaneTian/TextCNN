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
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_data (InputLayer)         [(None, 128)]        0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 128, 512)     1744384     input_data[0][0]                 
__________________________________________________________________________________________________
add_channel (Reshape)           (None, 128, 512, 1)  0           embedding[0][0]                  
__________________________________________________________________________________________________
convolution_3 (Conv2D)          (None, 126, 1, 128)  196736      add_channel[0][0]                
__________________________________________________________________________________________________
convolution_4 (Conv2D)          (None, 125, 1, 128)  262272      add_channel[0][0]                
__________________________________________________________________________________________________
convolution_5 (Conv2D)          (None, 124, 1, 128)  327808      add_channel[0][0]                
__________________________________________________________________________________________________
max_pooling_3 (MaxPooling2D)    (None, 1, 1, 128)    0           convolution_3[0][0]              
__________________________________________________________________________________________________
max_pooling_4 (MaxPooling2D)    (None, 1, 1, 128)    0           convolution_4[0][0]              
__________________________________________________________________________________________________
max_pooling_5 (MaxPooling2D)    (None, 1, 1, 128)    0           convolution_5[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 1, 384)    0           max_pooling_3[0][0]              
                                                                 max_pooling_4[0][0]              
                                                                 max_pooling_5[0][0]              
__________________________________________________________________________________________________
flatten (Flatten)               (None, 384)          0           concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 384)          0           flatten[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 18)           6930        dropout[0][0]                    
==================================================================================================
Total params: 2,538,130
Trainable params: 2,538,130
Non-trainable params: 0
__________________________________________________________________________________________________
```

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

| Loss | Accuracy | Val loss | Val accuracy |
| --- | --- | --- | --- |
| 0.1609 | 0.9683 | 0.3648 | 0.9185 |
### Test result
Use 2261 samples:

| Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | --- | --- | --- |
| 0.9363 | 0.9428 | 0.9310 | **0.9360** |
### Images
#### Accuracy
![Accuracy](https://github.com/ShaneTian/TextCNN/raw/master/acc.pdf)
#### Loss
![Loss](https://github.com/ShaneTian/TextCNN/raw/master/loss.pdf)
#### Confusion matrix
![Confusion matrix](https://github.com/ShaneTian/TextCNN/raw/master/confusion_matrix.pdf)

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
                        The fraction of test data.(default=0.1)
  -p PADDING_SIZE, --padding_size PADDING_SIZE
                        Padding size of sentences.(default=128)
  -e EMBED_SIZE, --embed_size EMBED_SIZE
                        Word embedding size.(default=512)
  -f FILTER_SIZES, --filter_sizes FILTER_SIZES
                        Convolution kernel sizes.(default=3,4,5)
  -n NUM_FILTERS, --num_filters NUM_FILTERS
                        Number of each convolution kernel.(default=128)
  -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                        Dropout rate in softmax layer.(default=0.5)
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of target classes.(default=18)
  -l REGULARIZERS_LAMBDA, --regularizers_lambda REGULARIZERS_LAMBDA
                        L2 regulation parameter.(default=0.01)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Mini-Batch size.(default=64)
  --epochs EPOCHS       Number of epochs.(default=10)
  --fraction_validation FRACTION_VALIDATION
                        The fraction of validation.(default=0.05)
  --results_dir RESULTS_DIR
                        The results dir including log, model, vocabulary and
                        some images.(default=./results/)
```

```
usage: test.py [-h] [-p PADDING_SIZE] [-c NUM_CLASSES] results_dir

This is the TextCNN test project.

positional arguments:
  results_dir           The results dir including log, model, vocabulary and
                        some images.

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
