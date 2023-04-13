# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import argparse
from random import randint

parser = argparse.ArgumentParser(
    prog='CUDA_DEV[ICE]=* main.py [run-all | train | test]',
    description='Train and test Torch models',
    epilog='Author: Yuxuan Zhang (zhangyuxuan@ufl.edu)')

parser.add_argument(
    '-m', '--model',
    type=str, required=True,
    help="Name of the model"
)

parser.add_argument(
    '-e', '--epochs',
    type=int, default=10,
    help="Number of epochs"
)

parser.add_argument(
    '-b', '--batch-size',
    type=int, default=10,
    help="Batch size"
)

# parser.add_argument(
#     '-l', '--lossFunction',
#     type=str, default="",
#     help="Loss function (not implemented yet)"
# )

parser.add_argument(
    '-r', '--learning-rate',
    type=float, default=1e-6,
    help="Learning rate"
)

# parser.add_argument(
#     '-k', '--kFoldRatio',
#     type=float, default=0.8,
#     help="Split ratio of k-Fold cross validation"
# )

parser.add_argument(
    '-L', '--load',
    type=str, default=None,
    help="Path to load pre-trained model"
)

parser.add_argument(
    '-s', '--seed',
    type=int, default=randint(0, 0xFFFFFFFF),
    help="Manually specify random seed, randomly generated if not specified"
)

parser.add_argument('command', nargs='*', type=str)

parser.add_help = True

if __name__ == "__main__":
    parser.print_help()
    print(parser.parse_args(['-m', 'model-name']))
else:
    ARGS = parser.parse_args()
    CMD = ARGS.command[0] if len(ARGS.command) else 'run-all'
    RUN_TRAIN = CMD in ['run-all', 'train']
    RUN_TEST = CMD in ['run-all', 'test']
    model: str = ARGS.model
    epochs: int = ARGS.epochs
    batch_size: int = ARGS.batch_size
    learning_rate: float = ARGS.learning_rate
    load: str | None = ARGS.load
    seed: int = ARGS.seed
    model = ARGS.model
