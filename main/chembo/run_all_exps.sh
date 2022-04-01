#!/bin/bash
# Runner of all experiments

export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/rdkit_contrib:${PWD}/synth/:${PWD}/synth/rexgen_direct"


n_exp_runs=$1

if [ -z $n_exp_runs ]
then
	echo "Provide the number of restarts"
	exit
fi

echo -e "\tQED+ChEMBL:"

# python experiments/run_chemist.py -d chembl -s 3 -o qed -b 100 -k distance_kernel_expsum -i 20 -stp 20 -mpl 1000
# python experiments/run_chemist.py -d chembl -s 3 -o qed -b 100 -k sum_kernel -i 20 -stp 20 -mpl 1000

python experiments/run_chemist.py -d chembl_small_qed -s 19 -o qed -b 50 -k distance_kernel_expsum -i 20 -stp 20 -mpl 1000

python experiments/run_explorer.py -d chembl -s 3 -o plogp -b 100 -i 20 -mpl 1000

for i in {1..$n_exp_runs}; do python experiments/run_chemist.py -d chembl_small_qed -s 19 -o qed -b 100 -k distance_kernel_expsum -i 20 -stp 20 -mpl 1000; done
for i in {1..$n_exp_runs}; do python experiments/run_chemist.py -d chembl -s 3 -o plogp -b 100 -k similarity_kernel -i 20 -stp 20 -mpl 1000; done
for i in {1..$n_exp_runs}; do python experiments/run_explorer.py -d chembl -s 3 -o plogp -b 100 -i 20 -mpl 1000; done

# echo -e "\tLogP+ChEMBL:" 

python experiments/run_explorer.py -d chembl_small_qed -s 19 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

python experiments/run_explorer.py -d chembl_large_qed -s 19 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

python experiments/run_explorer.py -d chembl -s 19 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000
# python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000

# # python experiments/run_chemist.py -d chembl_small_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# # python experiments/run_chemist.py -d chembl_small_qed -s 42 -o qed -b 100 -k distance_kernel_expsum -i 30 -stp 10 -mpl 1000

# # python experiments/run_chemist.py -d chembl_large_qed -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# # python experiments/run_chemist.py -d chembl_large_qed -s 42 -o qed -b 100 -k distance_kernel_expsum -i 30 -stp 10 -mpl 1000

# # python experiments/run_chemist.py -d chembl -s 42 -o qed -b 100 -k similarity_kernel -i 30 -stp 100 -mpl 1000
# # python experiments/run_chemist.py -d chembl -s 42 -o qed -b 100 -k distance_kernel_expsum -i 30 -stp 10 -mpl 1000

# echo -e "\tStarting running RandomExplorer"

# python experiments/run_explorer.py -d chembl_small_qed -s 19 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl_small_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

# python experiments/run_explorer.py -d chembl_large_qed -s 19 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl_large_qed -s 42 -o qed -b 100 -i 30 -mpl 1000

# python experiments/run_explorer.py -d chembl -s 19 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000
# # python experiments/run_explorer.py -d chembl -s 42 -o qed -b 100 -i 30 -mpl 1000

# # echo -e "\tStarting long RandomExplorer runs"
# # python experiments/run_explorer.py -d chembl -s 1 -o qed -b 1000 -i 30 -mpl None
# # python experiments/run_explorer.py -d chembl -s 3 -o qed -b 1000 -i 30 -mpl None




