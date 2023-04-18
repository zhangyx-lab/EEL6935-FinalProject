
from util.env import RUN_PATH, exists, relative
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('runID', nargs=1, type=str, help="Run ID")
parser.add_argument('command', nargs='*', type=str)
args = parser.parse_args()

RUN_ID = args.runID[0]
WORKDIR = RUN_PATH / RUN_ID

assert exists(WORKDIR), f"{relative(WORKDIR)} not exist"
