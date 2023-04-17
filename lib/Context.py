from datetime import datetime
from os.path import exists
from util.env import RUN_PATH, ensure
from shutil import rmtree
from math import floor, ceil
from lib.Signal import Signal
import util.run_log as run_log


def getRunID(start_i: int = 0):
    now = datetime.now().strftime("%Y%m%d")[2:]
    i = start_i
    while True:
        name = f"{now}-{i:02d}"
        if not exists(RUN_PATH / name):
            return name
        else:
            i += 1


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
            w = (80 - len(banner)) / 2
            if w < 1:
                print(banner)
            else:
                l = ''.join(['='] * ceil(w))
                r = ''.join(['='] * floor(w))
                print(l, banner, r)
        # Print args if applicable
        if len(args):
            with open(self.path / file, 'a') as log:
                print(*args, file=log)
            # Duplex to stdout
            if visible:
                print(f"{self.id} |", *args)

    def interrupt(self, code: int = -1):
        if self.parent is not None:
            self.parent.interrupt(code)

    __memory = {}

    def push(self, key, *values):
        """
        Push list of values into temp memory of current context.
        """
        if key in self.__memory:
            assert isinstance(self.__memory[key], list)
            for el in values:
                self.__memory[key].append(el)
        else:
            self.__memory[key] = [el for el in values]

    def collect(self, key, clear=True) -> list:
        """
        Collect all values previously pushed into memory.
        Removes collected set of value by default.
        """
        assert key in self.__memory, key
        mem = self.__memory[key]
        if clear:
            del self.__memory[key]
        return mem

    def collect_all(self, clear=True):
        def iterate_key(key: str):
            return key, self.collect(key, clear=clear)
        return map(iterate_key, list(self.__memory.keys()))

class Run(Context):
    state = None

    def __init__(self, id=getRunID()):
        super().__init__(id, RUN_PATH / id)
        self.signal.__enter__()
        self.signal.triggered = True
        self.log(banner=f"RUN ID: {id}")
        run_log.add(id)

    def context(self, context_name="train"):
        return Context(self.id, self.path / context_name, self)

    def interrupt(self, code: int = -1):
        self.log("")
        self.log(banner="Interrupted")
        self.state = code
        rmtree(self.path)
        run_log.remove(self.id)

    def __exit__(self, errType, err, traceback):
        if self.state is None and errType is None:
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            self.log(now, file="000_SUCCESS", banner="Finished")
            self.signal.__exit__()
        elif errType is not None and errType != SystemExit:
            print(errType)
            print(err)
            print(traceback)
            try:
                rmtree(self.path)
            except:
                pass
            run_log.remove(self.id, "aborted on error")


if __name__ == "__main__":
    with Run() as run:
        pass
