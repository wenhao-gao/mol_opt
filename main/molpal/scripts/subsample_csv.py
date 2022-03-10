import argparse
from functools import partial
import gzip
from pathlib import Path
import random
from typing import List

from tqdm import tqdm

def get_random_lines_from_file(filepath, title_line: bool = True,
                               k: int = 1, N: int = 1) -> List[List]:
    open_ = partial(gzip.open, mode='rt') if Path(filepath).suffix == '.gz' else open

    with open_(filepath) as fid:
        lines = [line for line in tqdm(fid)]
    
    liness = []
    for _ in range(N):
        if title_line:
            lines_ = [lines[0]]
            lines_.extend(random.choices(lines[1:], k=k))
        else:
            lines_ = random.choices(lines, k=k)
        liness.append(lines_)

    return liness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--no-title-line', action='store_true', default=False)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-N', type=int, default=1)
    args = parser.parse_args()

    print(args, flush=True)

    liness = get_random_lines_from_file(
        args.input, not args.no_title_line, args.k, args.N
    )

    for i, lines in enumerate(liness):
        with open(f'{args.output}_{i}.csv', 'w') as fid:
            fid.writelines(lines)

if __name__ == '__main__':
    main()