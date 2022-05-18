
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plots the validation loss from the logs.')
    parser.add_argument('name_path_pairs', type=str, nargs='+',
                        help='name for plot and the path to log')
    args = parser.parse_args()
    name_paths = args.name_path_pairs


    regex = re.compile("V__([-+]?\d*\.\d+|\d+)__([-+]?\d*\.\d+|\d+)")

    line_styles = [':', '--', '-.', '-']
    with plt.style.context('bmh'):
        f, ax = plt.subplots(figsize=(10,6))
        for i in range(len(name_paths) // 2):
            name = name_paths[2*i]
            path = name_paths[2*i + 1]
            with open(path, 'r') as fo:
                ls_ = '\n'.join(fo.readlines())
            all_vals = regex.findall(ls_)
            loss_adj = np.array([float(x[1]) for x in all_vals])
            indics = np.arange(len(loss_adj))
            plt.plot(indics, loss_adj, line_styles.pop(), lw=1.5, label=name)
        plt.xlabel('epoch')
        plt.ylabel('Normalized MAE on validation set')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='gray', linestyle='--')
        plt.legend()
        plt.savefig('qm9_training_val_loss.png')


if __name__ == '__main__':
    main()
