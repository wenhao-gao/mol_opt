#!/bin/bash
#SBATCH -J molpal
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=SIGTERM@120        # send a SIGTERM 120s before timing out

#SBATCH -N 1                        # number of nodes
#SBATCH --ntasks-per-node 1
#SBATCH -c 8                        # cores per task

#SBATCH --mem-per-cpu 4000                 # total memory
#SBATCH -t 0-08:00
#SBATCH -p normal      # Partition to submit to

config=$1

source activate molpal

export NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}' )

######################## DO NOT CHANGE THINGS HERE ############################
redis_password=$( uuidgen 2> /dev/null )
export redis_password

nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST ) # Getting the node names
nodes_array=( $nodes )

node_0=${nodes_array[0]} 
ip=$( srun -N 1 -n 1 -w $node_0 hostname --ip-address ) # making redis-address
port=$( python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()' )
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

srun -N 1 -n 1 -w $node_0 ray start --head \
    --node-ip-address=$ip --port=$port --redis-password=$redis_password \
    --num-cpus $SLURM_CPUS_ON_NODE --num-gpus $NUM_GPUS \
    --temp-dir /tmp/degraff --block > /dev/null 2>& 1 &
sleep 30

worker_num=$(( $SLURM_JOB_NUM_NODES - 1 ))
for ((  i=1; i<=$worker_num; i++ )); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    srun -N 1 -n 1 -w $node_i ray start --address $ip_head \
        --redis-password=$redis_password \
        --num-cpus $SLURM_CPUS_ON_NODE --num-gpus $NUM_GPUS \
        --temp-dir /tmp/degraff --block > /dev/null 2>& 1 &
    sleep 5
done
###############################################################################

python run.py --config $config --ncpu $SLURM_CPUS_PER_TASK