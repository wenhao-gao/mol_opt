import signal
import sys
from timeit import default_timer as time

from main.molpal.molpal import args, Explorer

def sigterm_handler(signum, frame):
    sys.exit(0)
signal.signal(signal.SIGTERM, sigterm_handler)

def main():
    
    params = vars(args.gen_args())

    path = params.pop("output_dir")
    explorer = Explorer(path, **params)
    explorer.run()


if __name__ == "__main__":
    main()