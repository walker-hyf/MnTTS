# MnTTS: An Open-Source Mongolian Text-to-Speech Synthesis Dataset and Accompanied Baseline
 
## Introduction
This is an implementation of the following paper.
> [MnTTS: An Open-Source Mongolian Text-to-Speech Synthesis Dataset and Accompanied Baseline.](https://arxiv.org/abs/2209.10848)
> in Proc. IALP'2022

Yifan Hu, Pengkai Yin, [Rui Liu *](https://ttslr.github.io/), Feilong Bao, Guanglai Gao.
 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.10848)

## 0) Environment Preparation

This project uses `conda` to manage all the dependencies, you should install [anaconda](https://anaconda.org/) if you have not done so. 

```bash
# Clone the repo
git clone https://github.com/walker-hyf/MnTTS.git
cd $PROJECT_ROOT_DIR
```

### Install dependencies
```bash
conda env create -f Environment/environment.yaml
```

### Activate the installed environment
```bash
conda activate mntts
```

## 1) Prepare MnTTS Dataset

Prepare our MnTTS dataset in the following format:
```
|- MnTTS/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```

Where `metadata.csv` has the following format: `id|transcription`. This is a ljspeech-like format.

[The complete dataset is available from our multilingual corpus website](http://mglip.com/corpus/corpus_detail.html?corpusid=20220819185345).


## 2) Preprocessing

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

To reproduce the steps above:
```
tensorflow-tts-preprocess --rootdir ./MnTTS --outdir ./dump_mntts --config preprocess/mntts_preprocess.yaml --dataset mntts
```

```
tensorflow-tts-normalize --rootdir ./dump_mntts --outdir ./dump_mntts --config preprocess/mntts_preprocess.yaml --dataset mntts
```

 



## 3) Training TacoTron2 from scratch with MnTTS dataset

Based on the script [`train_tacotron2.py`](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/tacotron2/train_tacotron2.py).

 
This example code show you how to train Tactron-2 from scratch with Tensorflow 2 based on custom training loop and tf.function. 

  
Here is an example command line to training TacoTron2 from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train-dir ./dump_mntts/train/ \
  --dev-dir ./dump_mntts/valid/ \
  --outdir ./examples/tacotron2/exp/train.tacotron2.v1/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode.

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/ckpt-100000
```

If you want to finetune a model, use `--pretrained` like this with your model filename
```bash
--pretrained pretrained.h5
```

Extract duration from alignments for FastSpeech

You may need to extract durations for student models like fastspeech. Here we use teacher forcing with window masking trick to extract durations from alignment maps:

Extract for valid set:

```
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump_mntts/valid/ \
  --outdir ./dump_mntts/valid/durations/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-100000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32
  --win-front 3 \
  --win-back 3
```

Extract for training set:

```
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump_mntts/train/ \
  --outdir ./dump_mntts/train/durations/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-100000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32
  --win-front 3 \
  --win-back 3
```

To extract postnets for training vocoder, follow above steps but with `extract_postnets.py`

## 4) Training FastSpeech2 from scratch with MnTTS dataset

Based on the script [`train_fastspeech2.py`](https://github.com/dathudeptrai/TensorFlowTTS/blob/master/examples/fastspeech2/train_fastspeech2.py).

Here is an example command line to training FastSpeech2 from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump_mntts/train/ \
  --dev-dir ./dump_mntts/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump_mntts/stats_f0.npy \
  --energy-stat ./dump_mntts/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""
```


## 5) Vocoder Training


First, you need training generator with only stft loss:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump_mntts/train/ \
  --dev-dir ./dump_mntts/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:


```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump_mntts/train/ \
  --dev-dir ./dump_mntts/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --resume ./examples/hifigan/exp/train.hifigan.v1/checkpoints/ckpt-100000
```

## 6) MnTTS Model Inference

You can follow below example command line to generate synthesized speech for given text in 'dump_mntts/inference.txt' using Griffin-Lim and trained HiFi-GAN vocoder:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/inference_fastspeech2.py \
    --outdir prediction/MnTTS_inference \
    --infile dump_mntts/inference.txt  \
    --tts_ckpt examples/fastspeech2/exp/train.fastspeech2.v1/checkpoints/model-200000.h5 \
    --vocoder_ckpt  examples/hifigan/exp/train.hifigan.v1/checkpoints/generator-200000.h5 \
    --stats_path dump_mntts/stats.npy \
    --dataset_config preprocess/mntts_preprocess.yaml \
    --tts_config examples/fastspeech2/conf/fastspeech2.v1.yaml \
    --vocoder_config examples/hifigan/conf/hifigan.v1.yaml \
    --lan_json dump_mntts/mntts_mapper.json 
```

You can find pre-trained models in the [Links](#Links) section.


The synthesized speech will save to `prediction/MnTTS_inference` folder.


## Links

- Pre-trained models: [download pre-trained models](https://drive.google.com/file/d/1eVtGQvRd7UKAEHOCricQ5RSAgminoCd_/view?usp=sharing)
- Demo: [link to synthesized audio samples](https://github.com/walker-hyf/MnTTS/tree/main/prediction/MnTTS_inference)


[//]: # (## Citation)
[//]: # (Please kindly cite the following paper if you use this code repository in your work,)
[//]: # (```)
[//]: # (```)


## Author

E-mail：hyfwalker@163.com

## Acknowledgements:


Tensorflow-TTS: [https://github.com/TensorSpeech/TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)


