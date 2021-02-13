import argparse
import csv
import math
import glob

import librosa
import pandas as pd

from subcommands.create_samples import create_samples
from subcommands.classify_file import classify_file
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonumator")
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands')

    samples_subparser = subparsers.add_parser('create_samples')
    samples_subparser.add_argument('--path', required=True, help='the path to the .wav files to be processed')
    samples_subparser.add_argument('--csv', required=True, help='the .csv file that contains the start and end times for sampling')
    samples_subparser.add_argument('--output', default="output/", help='the output directory to write files to')
    samples_subparser.set_defaults(func=create_samples)

    classify_subparser = subparsers.add_parser('classify_file')
    classify_subparser.add_argument('--file', required=True, help='the path to the .wav file to be processed')
    classify_subparser.add_argument('--training-set', required=True, help='the path to the training set to')
    classify_subparser.add_argument('--output', required=True, help='the output path for the generated .csv file')
    classify_subparser.set_defaults(func=classify_file)

    args = parser.parse_args()
    args.func(args)