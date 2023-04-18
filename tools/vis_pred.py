import cv2
from argparse import ArgumentParser
from util.env import RUN_PATH, exists, relative
from util.loader import train_data, test_data

parser = ArgumentParser()
parser.add_argument('runID', nargs=1, type=str)
parser.add_argument('command', nargs='*', type=str)
args = parser.parse_args()

WORKDIR = RUN_PATH / args.runID

assert exists(WORKDIR), relative(WORKDIR)

