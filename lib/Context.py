from sys import argv
from datetime import datetime
from os.path import exists
from util.env import RUN_PATH, VAR_PATH, ensure
from shutil import rmtree
from math import floor, ceil
from lib.Signal import Signal


def getRunID(start_i: int = 0):
    now = datetime.now().strftime("%Y%m%d")[2:]
    i = start_i
    while True:
        name = f"{now}-{i:02d}"
        if not exists(RUN_PATH / name):
            return name
        else:
            i += 1


RUN_LOG_PATH = RUN_PATH / "log.txt"
RUN_LOG_PATH.touch(exist_ok=True)


class Context:
    def __init__(self, id, path, parent=None):
        ensure(path)
        self.id = id
        self.path = path
        self.parent = parent
        self.signal = Signal(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log(self, *args, file="log.txt", banner=None, visible=True):
        if len(args) == 0 and banner is None:
            def p(*args):
                if len(args) == 0:
                    return
                self.log(*args, file=file)
            return p
        # Print banner (optional)
        if banner is not None:
            w = (60 - len(banner)) / 2
            if w < 1:
                print(banner)
            else:
                l = ''.join(['='] * ceil(w))
                r = ''.join(['='] * floor(w))
                print(l, banner, r)
        # Print args if applicable
        if len(args) and visible:
            with open(self.path / file, 'a') as log:
                print(*args, file=log)
            # Duplex to stdout
            print(f"{self.id} |", *args)

    def interrupt(self, code: int = -1):
        if self.parent is not None:
            self.parent.interrupt(code)


class Run(Context):
    state = None

    def __init__(self, id=getRunID()):
        super().__init__(id, RUN_PATH / id)
        with open(RUN_LOG_PATH, "a") as f:
            print(f"{id} | {' '.join(argv)}", file=f)
        self.log(banner=f"RUN ID: {id}")

    def context(self, context_name="train"):
        return Context(self.id, self.path / context_name, self)

    def interrupt(self, code: int = -1):
        self.state = code
        rmtree(self.path)

    def __exit__(self, errType, err, traceback):
        if self.state is None and errType is None:
            now = datetime.now().strftime("%Y%m%d %H:%M:%S")
            self.log(now, file="000_SUCCESS")


if __name__ == "__main__":
    with Run() as run:
        pass
