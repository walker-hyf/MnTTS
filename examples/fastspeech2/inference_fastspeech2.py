# import glob

import time, os
import argparse
import yaml
import soundfile as sf
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow_tts.utils import TFGriffinLim, griffin_lim_lb
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

def tts_inference(args, ttsmodel, vocoder, processor):
    with open(args.infile, 'r') as f:
        for line in f:
            mon_text  = line.split('|')[1].strip()
            input_ids = processor.text_to_sequence(mon_text)
            mel_before, mel_after, duration_outputs, _, _ = ttsmodel.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
            )
            filename = line.split('|')[0].strip()
            syn_gl(args, mel_after, filename)
            syn_vocoder(args, vocoder, mel_after,  filename)


def syn_gl(args, mel, filename):
    # Single file
    mel = mel.numpy()
    mel = mel[0]
    print(mel.shape)  # [len, 80]
    config = yaml.load(open(args.dataset_config), Loader=yaml.Loader)
    griffin_lim_lb(mel, args.stats_path, config, 32, args.outdir, wav_name=filename + '-gl')


def syn_vocoder(args, vocoder, mel, filename):
    audio_after = vocoder.inference(mel)[0, :, 0]
    # save to file
    sf.write(os.path.join(args.outdir, filename + '-vocoder.wav'), audio_after, 22050, "PCM_16")


def main():
    """Running decode fastspeech2 mel-spectrogram."""
    parser = argparse.ArgumentParser(
        description="Inference with fastspeeech2"
    )

    parser.add_argument(
        "--outdir",
        default="prediction/MnTTS_inference",
        type=str, required=True, help="directory to save generated speech."
    )
    parser.add_argument(
        "--infile",
        default="dump_mntts/inference.txt",
        type=str, required=True, help="inference text."
    )
    parser.add_argument(
        "--tts_ckpt",
        default="examples/fastspeech2/exp/train.fastspeech2.v1/checkpoints/model-200000.h5",
        type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--vocoder_ckpt",
        default="examples/hifigan/exp/train.hifigan.v1/checkpoints/generator-200000.h5",
        type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--stats_path",
        default="dump_mntts/stats.npy",
        type=str,
        required=True,
        help="stats path",
    )
    parser.add_argument(
        "--dataset_config",
        default="preprocess/mntts_preprocess.yaml",
        type=str,
        required=True,
        help="dataset_config path",
    )
    parser.add_argument(
        "--tts_config",
        default='examples/fastspeech2/conf/fastspeech2.v1.yaml',
        type=str,
        required=True,
        help="tts_config path",
    )
    parser.add_argument(
        "--vocoder_config",
        default='examples/hifigan/conf/hifigan.v1.yaml',
        type=str,
        required=True,
        help="vocoder_config path",
    )
    parser.add_argument(
        "--lan_json",
        default="dump_mntts/mntts_mapper.json",
        type=str,
        required=True,
        help="language json  path",
    )
    args = parser.parse_args()

    # initialize fastspeech2 model.
    tts_config = AutoConfig.from_pretrained(args.tts_config)
    ttsmodel = TFAutoModel.from_pretrained(
        config=tts_config,
        pretrained_path=args.tts_ckpt
    )

    # initialize HiFi-GAN model
    vocoder_config = AutoConfig.from_pretrained(args.vocoder_config)
    vocoder = TFAutoModel.from_pretrained(
        config=vocoder_config,
        pretrained_path=args.vocoder_ckpt
    )

    processor = AutoProcessor.from_pretrained(pretrained_path=args.lan_json)

    tts_inference(args,ttsmodel, vocoder, processor)
    print('ok')


if __name__ == '__main__':
    main()