import math
import os
import glob

import librosa
import pandas as pd
import soundfile as sf

def between(start, end):
    if start <= number <= end:
        return True
    else:
        return False

def make_sample(file_list, start_time, end_time, padding=0):
    """
    Each file in `file_list` is 4 hours long, calculate the offset in the file
    based on the relative time.
    """
    index = math.floor(start_time / 14400)

    indexed_start_time = start_time - (index * 14400)
    indexed_end_time = end_time - (index * 14400) + padding

    return librosa.load(
        file_list[index],
        offset=indexed_start_time,
        sr=5000,
        duration=indexed_end_time - indexed_start_time,
    )


def create_damselfish_samples(df, file_list, output):
    os.mkdir(output)
    for i in range(0, len(df)):
        row = df.iloc[i]
        sample, sr = make_sample(file_list, row[0], row[1], .5)
        sf.write(f"{output}/{i}.wav", sample, sr)


def create_noise_samples(df, file_list, output):
    os.mkdir(output)
    for i in range(0, len(df)):
        sample, sr = make_sample(file_list, start_time + i, start_time + i + 1, .5)
        sf.write(f"{output}/{i}.wav", sample, sr)


def create_samples(args):
    df = pd.read_csv(args.csv, usecols=[3, 4])
    file_list = glob.glob(f"{args.path}/*.wav")
    file_list.sort()

    create_damselfish_samples(df, file_list, f"{args.output}/damselfish")
    create_noise_samples(df, file_list, f"{args.output}/noise")
