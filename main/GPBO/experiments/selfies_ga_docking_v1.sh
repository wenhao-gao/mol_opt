#!/usr/bin/env bash
# Graph GA on guacamol dataset (of varying sizes)
objective_arr=( \
"F2_qed-pen-v3"  \
"JAK2-not-LCK-v2_qed-pen-v3" \
)
dataset_size_arr=( 100 1000 10000 100000 )
method_name="selfies_ga"

# These must match...
# ORIG
max_func_calls_arr=( 100 1000 10000 )
max_func_calls_arr=( 100 1000 )  # TODO!!
offspring_size_arr=(   1    1    10 )

# logging
log_dir="./results/ai4sci/log/"
mkdir -p "$log_dir"

curr_expt_idx=0
for target in "${objective_arr[@]}" ; do

    # Result dir for this target
    res_dir="./results/ai4sci/res/${method_name}/${target}"
    mkdir -p "${res_dir}"

    # max func calls
    for budget_i in "${!max_func_calls_arr[@]}" ; do 
        max_func_calls="${max_func_calls_arr[budget_i]}"
        offspring_size="${offspring_size_arr[budget_i]}"

        # Dataset size
        for dataset_size in "${dataset_size_arr[@]}" ; do

            # Multiple trials
            for trial in {0..2}; do
                expt_id_str="budget-${max_func_calls}_Ndata-${dataset_size}_trial-${trial}"
                output_path="${res_dir}/${expt_id_str}.json" 
                dataset_path="./data/dockstring-splits/seed_${trial}/rand_${dataset_size}.tsv"

                if [[ -z "$expt_idx" || "$expt_idx" = "$curr_expt_idx" ]] ; then

                    if [[ -f "$output_path" ]]; then
                        echo "Results for expt_idx ${curr_expt_idx} ${method_name} ${target} ${expt_id_str} exists! Skipping."
                    else


                        echo "Running expt_idx ${curr_expt_idx} ${method_name} ${target} ${expt_id_str}..."

                        PYTHONPATH="$(pwd)/src:$PYTHONPATH" python src/mol_opt/run_${method_name}.py \
                            --dataset="$dataset_path" \
                            --objective="${target}" \
                            --num_cpu=8 \
                            \
                            --max_func_calls="${max_func_calls}" \
                            --offspring_size="$offspring_size" \
                            \
                            --output_path="${output_path}" \
                            &> "${log_dir}/${method_name}_${target}_${expt_id_str}.log"
                    
                    fi
                fi

                # Increment experiment index after every potential experiment
                curr_expt_idx=$(( curr_expt_idx + 1 ))

            done
        done
    done
done
