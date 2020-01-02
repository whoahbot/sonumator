import argparse
import csv
import math
import glob

import librosa
import pandas as pd

from subcommands.create_samples import create_samples
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonumator")
    parser.add_argument('--path', required=True, help='the path to the .wav files to be processed')
    parser.add_argument('--csv', required=True, help='the .csv file that contains the start and end times for sampling')
    parser.add_argument('--output', default="output/", help='the output directory to write files to')

    args = parser.parse_args()
    create_samples(args)