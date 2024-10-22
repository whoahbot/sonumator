import math
import os
import glob

import librosa
import pandas as pd
import soundfile as sf
import random

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
        duration=2,
    )


def create_damselfish_samples(df, file_list, output):
    try:
        os.mkdir(output)
    except OSError:
        pass
    for i in range(0, len(df)):
        row = df.iloc[i]
        sample, sr = make_sample(file_list, row[0], row[1], .5)
        sf.write(f"{output}/{i}.wav", sample, sr)


def random_start(df):
    """
    Find a random start position that isn't between any of
    the start and stop points in df.
    """
    beginning = 48 # After the end of the first sample
    end = df.iloc[(len(df) -1)][1] # The last entry in the spreadsheet
    
    start = random.uniform(beginning, end)
    for i in range(0, len(df)):
        row = df.iloc[i]
        if row[0] < start < row[1]:
            random_start(df)
    
    return start


def create_noise_samples(df, file_list, output):
    try:
        os.mkdir(output)
    except OSError:
        pass
        
    for i in range(0, len(df)):
        start_time = random_start(df)
        sample, sr = make_sample(file_list, start_time + i, start_time + i + 1, .5)
        sf.write(f"{output}/{i}.wav", sample, sr)


def create_samples(args):
    df = pd.read_csv(args.csv, usecols=[3, 4])
    file_list = glob.glob(f"{args.path}/*.wav")
    file_list.sort()

    create_damselfish_samples(df, file_list, f"{args.output}/damselfish")
    create_noise_samples(df, file_list, f"{args.output}/noise")
