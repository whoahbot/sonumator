import math
import glob

import librosa
import pandas as pd
import soundfile as sf

def make_sample(file_list, start_time, end_time):
    """
    Each file in `file_list` is 4 hours long, calculate the offset in the file
    based on the relative time. The time index for each file was off by 6 seconds
    so add that for each file by index too.
    """
    index = math.floor(start_time/14400)
      
    indexed_start_time = start_time - (index * 14400) + (index * 6)
    indexed_end_time = end_time - (index * 14400) + (index * 6)

    return librosa.load(file_list[index], offset=indexed_start_time, sr=5000, duration=indexed_end_time - indexed_start_time)


def create_samples(args):
    df = pd.read_csv(args.csv, usecols=[3,4])
    file_list = glob.glob(f"{args.path}/*.wav")
    file_list.sort()

    for i in range(0, 100):
        row = df.iloc[i]
        sample, sr = make_sample(file_list, row[0], row[1])
        sf.write(f"{args.output}/{i}.wav", sample, sr)
    