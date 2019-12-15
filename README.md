# keras_insightFace

The implementation referred to [the official implementation in mxnet](https://github.com/deepinsight/insightface)

## TODO List

1. *reimplementation fmobilefacenet from official implementation in mxnet [done!]*
2. *Train model use multiple gpu [done!]*
2. *Train fmobilefacenet with softmax [done!]*
3. *Model evaluation [done!]*
4. *convert model to tensorflow-lite [done!]*
4. *deploy in Android [done!]*
5. *training fmobilefacenet with mrigin and meet official standards in acc [doing!]*
6. *reimplementation fmnasnet from official implementation in mxnet [to do]*

7. Backbones    
   7.1 *fmobilefacenet [done!]*    
   7.2 **fmnasnet [todo]**    
8. Losses
   8.1 *Softmax loss [done!]*    
   8.1 *Arcface loss [done!]*    
   8.2 **Cosface loss [done]**    
   8.3 **Sphereface loss [done]**    
   8.4 **Triplet loss [todo]**
9.  **Face detection and alignment [todo]**

## Running Environment

- python 3.6 
- tensorflow 1.14.0+


### Pretrained Model

Pretrained models and their accuracies on validation datasets are shown as following:

fmobilefacenet acc:

|losses|lfw|agedb_30|download url|
|:----:|:----:|:----:|:----:|
|Softmax loss|0.9923%|93.78%|[baidu](https://pan.baidu.com/s/1NHkQ3WhvHTotyVnfCX_rJw) 密码: a4fj|
|Margin loss|waiting to train|waiting to train||
|Arcface loss|waiting to train|waiting to train||

before use the download model to evalua, please convert gpu model to single [convert_to_single_gpu_inference.py] gpu first 

### Model Evaluation

You can evaluate a pretrained model with [evaluate.py](https://github.com/luckycallor/InsightFace-tensorflow/blob/master/evaluate.py) by specifying the config path and model path, for example:

```
python evaluate.py 
--config_path=./configs/config_ms1m_100.yaml 
--model_path=path/to/model
```

## Train

training model in your computer

### Data Prepare

The official InsightFace project open their training data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). This data is in mxrec format, you can transform it to tfrecord format with [./data/generateTFRecord.py](https://github.com/luckycallor/InsightFace-tensorflow/blob/master/data/generateTFRecord.py) by the following script:

```
python tools/generate_images_from_mxnet 
--mxnet_path=$DIRECTORY_TO_THE_TRAINING_DATA$
--save_dir=$DIRECTORY_TO_THE_TRAINING_DATA$
```

### Starting train

begin start to train, please replace configs/fmobilefacenet_ms1m.yaml gpus option set to your PC

```
python train.py --config_path=./configs/fmobilefacenet_ms1m.yaml
```


## Android deploy 

will make it can use soon...