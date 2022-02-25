"""
Plot optimization paths from logs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
font = {#'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)


# VIS_DIR = 'experiments/results/visualizations'
VIS_DIR = 'experiments/visualizations'


def plot_paths(name, grouped_paths_list, labels):
    results = []
    for paths_list in grouped_paths_list:
        group = []
        for path in paths_list:
            res = get_list_from_file(path)
            group.append(res)
        results.append(group)

    # preprocessing
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)

    max_len = max([max([len(res) for res in group])
                    for group in results])

    for i, (group, label) in enumerate(zip(results, labels)):
        for res in group:
            res.extend( [res[-1]] * (max_len - len(res)) )
        avg = np.array(group).mean(axis=0)
        stddev = np.array(group).std(axis=0) / np.sqrt(5)  # SE correction
        plt.plot(range(20, 20 + len(avg)), avg, label=label)
        plt.fill_between(range(20, 20 + len(avg)), avg-stddev, avg+stddev, alpha=0.1)

    # plot eps
    # plt.title("Penalized logP optimization", fontsize=32)
    plt.title("QED optimization", fontsize=32)
    plt.xlabel("BO iteration", fontsize=28)
    plt.ylabel("qed value", fontsize=28)
    plt.ylim(0.82, 0.95)
    # plt.ylim(0.6, 0.95) # for low-value QED
    # plt.ylim(1.6, 12.5)
    plt.legend(loc='lower right', prop={'size': 32}) #'upper left'
    plot_path = os.path.join(VIS_DIR, f"{name}.pdf")
    # plt.show()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent.x0 -= 1.1
    extent.y0 -= 1.
    extent.y1 += 0.9
    plt.savefig(plot_path, bbox_inches=extent, pad_inches=0)
    # plt.savefig(plot_path, format='eps', dpi=1000)

    plt.clf()

def get_list_from_file(path):
    res = []
    with open(path, 'r') as f:
        if path.endswith('exp_log'):
            # Chemist
            for line in f:
                if line.startswith("#"):
                    curr_max = line.split()[3]
                    curr_max = float(curr_max.split("=")[1][:-1])
                    res.append(curr_max)
        elif path.endswith('opt_vals'):
            # Explorer
            line = f.readline()
            res = [float(v) for v in line.split()]
    return res

def format_chem(group, dir="final"):
    return [f"./experiments/results/{dir}/chemist_exp_dir_{exp_num}/exp_log" for exp_num in group]

def format_rand(group):
    return [f"./experiments/results/final/rand_exp_dir_{exp_num}/opt_vals" for exp_num in group]


if __name__ == "__main__":
    # Random starting value
        qed_paths_list = [
        format_rand(["20190522072706", "20190522072835", "20190522073130", "20190522073258", "20190522160909"]),
        format_chem(["20190518132128", "20190518184219", "20190519053538", "20190519172351", "20190522161028"]),
        format_chem(["20190518095359", "20190518051956", "20190518182118", "20190518182227", "20190520042603"]),
        format_chem(["20190929152544", "20191001135403", "20191004120521", "20190929152544", "20191004120901"],  dir="sum_kernel")
    ]

    plogp_paths_list = [
        format_rand(["20190522072622", "20190522072737", "20190522072853", "20190522154201", "20190522154310"]),
        format_chem(["20190519053341", "20190520035241", "20190520051810", "20190520123558", "20190520034409"]),
        format_chem(["20190520034402", "20190520034422", "20190520041405", "20190518051956_f", "20190520034409"]),
        format_chem(["20191001225737", "20191001225800", "20191001225748", "20191004120702", "20191004120851"],  dir="sum_kernel")
    ]

    # # Low starting value
    # qed_paths_list = [
    # format_rand(["20190627233317", "20190627233700", "20190627233943", "20190627234154", "20190627234353"]),
    # format_chem(["20190622011502", "20190622012453", "20190626002756", "20190627234558", "20190704180148"]),
    # format_chem(["20190622012539", "20190626002955", "20190626002927", "20190627233542", "20190630230013"])
    #         ]

    # plogp_paths_list = [
    #             format_rand(["20190627233406", "20190627233717", "20190707151154", "20190707150845", "20190707151321"]),
    #             format_chem(["20190630225902", "20190707150454", "20190707150643", "20190707150736", "20190704180148"]),
    #             format_chem(["20190707150242", "20190707150547", "20190707150515", "20190627233542", "20190630230013"])
    #         ]

    name = "qed_result_with_sum_2"
    # name = "qed_result_with_sum"
    labels = ["rand", "fingerprint", "OT-dist", "sum-kernel"]
    plot_paths(name, qed_paths_list, labels)

