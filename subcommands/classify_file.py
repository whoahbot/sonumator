import os
import math
import glob
import time
import concurrent.futures

from subprocess import call, check_output
from dateutil.parser import parse
from datetime import timedelta

import matplotlib.pyplot as plt
import librosa
import torchaudio
import librosa.display
import pandas as pd
import soundfile as sf

from fastai_audio.audio import *
from fastai_audio.audio import SpectrogramConfig, AudioConfig
from fastai.vision import models
from fastai.vision import *


def fetch_recording(gs_path):
    return_code = call(["gsutil", "cp", gs_path, "recordings/"])
    print(return_code)


def list_all_recordings():
    recordings = (
        check_output(["gsutil", "ls", "gs://sonumator/recordings/2015-2016/*.wav"])
        .decode("utf-8")
        .splitlines()
    )
    recordings.sort()

    return recordings


def search_file_for_samples(gs_filepath, learn, output, offset=2):
    fetch_recording(gs_filepath)
    filepath = f"recordings/{os.path.basename(gs_filepath)}"
    df = pd.DataFrame(columns=["start", "end", "filepath"])

    si, ei = torchaudio.info(filepath)
    length = si.length / si.rate

    for i in range(0, int(length), offset):
        end = i + offset

        basename = os.path.basename(filepath)[:-4]
        filetime = parse(basename)

        y, sr = librosa.load(filepath, sr=5000, offset=i, duration=2)
        tmpfile = f"potentials/{basename}-{i}.wav"
        sf.write(tmpfile, y, 5000)
        item = AudioItem(path=tmpfile)
        category, _, _ = audio_predict(learn, item)
        if str(category) == "damselfish":
            df = df.append(
                [
                    {
                        "start": filetime + timedelta(seconds=i),
                        "end": filetime + timedelta(seconds=end),
                        "filepath": filepath,
                    }
                ]
            )
        os.remove(tmpfile)

    os.remove(filepath)
    output_csv = f"{output}/{basename}.csv"
    df.to_csv(output_csv)

    # Copy the file to the GS bucket
    return_code = call(["gsutil", "cp", output_csv, "gs://sonumator/csv/"])
    print(f"Copied {output_csv} to google storage, return code: {return_code}")


def classify_file(args):
    # Set a seed for reproducability
    torch.manual_seed(0)
    path = Path(args.training_set)

    sg_cfg = SpectrogramConfig(
        f_min=200.0,
        f_max=1000.0,
        hop_length=32,
        n_fft=128,
        n_mels=64,
        pad=0,
        win_length=None,
    )

    config = AudioConfig(
        use_spectro=True,
        # delta=True,
        sg_cfg=sg_cfg,
    )

    al = (
        AudioList.from_folder(path, config=config)
        .split_by_rand_pct(0.2, seed=4)
        .label_from_folder()
    )

    tfms = None
    # tfms = get_spectro_transforms(mask_time=False, mask_freq=True, roll=False, num_rows=12)
    tfms = get_spectro_transforms(
        size=(128, 626),  # Upscale the spectrograms from 64x313
        mask_frequency=False,  # Don't mask frequencies
        mask_time=False,  # Don't mask time
    )
    db = al.transform(tfms).databunch(bs=10)

    learn = audio_learner(db, base_arch=models.resnet50)
    learn = learn.load("weight_decay_more_data_465")
    with torch.no_grad():
      learn.model.eval()

      search_file_for_samples(args.file, learn, args.output)
