import csv
import logging

from pydub import AudioSegment

def create_clip(segment, start, end):
    # pydub works in ms
    begin = float(start) * 1000
    end = float(end) * 1000

    print(begin, end)

    return segment[begin:end]


def find_samples(csv_file):
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        return [(row['Begin Time (s)'], row['End Time (s)']) for row in reader]

if __name__ == "__main__":
    segment = AudioSegment.from_wav("20150615-003510.wav")

    timecodes = find_samples('damsel_samples.csv')
    just_a_few = timecodes[:6]

    for idx, timecode in enumerate(just_a_few):
        clip = create_clip(segment, timecode[0], timecode[1])
        clip.export(f"sample_{idx}.wav")
