import argparse
import csv
import math
import glob

import librosa
import pandas as pd

from subcommands.create_samples import create_samples
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonumator")
    parser.add_argument('--path', help='the path to the .wav files to be processed')
    parser.add_argument('--csv', help='the .csv file that contains the start and end times for sampling')

    args = parser.parse_args()
    create_samples(args)