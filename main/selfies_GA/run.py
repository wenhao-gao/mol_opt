import os, time
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from selfies import decoder, encoder 
import multiprocessing
import torch
import discriminator as D
import evolution_functions as evo
import generation_props as gen_func
import numpy as np 
import argparse, yaml  
from main.optimizer import BaseOptimizer 
from tdc import Oracle


class SELFIES_GA_optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "selfies_ga"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        beta = config['beta']
        max_generations = config['max_generations'] ## 1000
        generation_size = config['generation_size'] ##  500

        results_dir = evo.make_clean_results_dir()
        i = 0 
        max_fitness_collector = []
        image_dir, saved_models_dir, data_dir = evo.make_clean_directories(beta, results_dir, i) # clear directories 

        torch.cuda.empty_cache()


        starting_selfies = [encoder('C')]
        max_molecules_len = 81
        disc_epochs_per_generation = 10
        disc_enc_type = 'properties_rdkit'    # 'selfies' or 'smiles' or 'properties_rdkit'
        disc_layers = [100, 10]
        training_start_gen = 0   # generation index to start training discriminator
        device = 'cpu'
        num_processors = multiprocessing.cpu_count()
        impose_time_adapted_pen = True

        
        # Obtain starting molecule
        starting_smiles = evo.sanitize_multiple_smiles([decoder(selfie) for selfie in starting_selfies])
        
        # Recording Collective results
        smiles_all         = []    # all SMILES seen in all generations
        selfies_all        = []    # all SELFIES seen in all generation
        smiles_all_counter = {}    # Number of times a SMILE string is recorded in GA run
        
        
        # Initialize a Discriminator
        discriminator, d_optimizer, d_loss_func = D.obtain_initial_discriminator(disc_enc_type, disc_layers, max_molecules_len, device)
        
        # Read in the Zinc data set 
        molecules_reference = evo.read_dataset_encoding(disc_enc_type)
        molecules_reference = dict.fromkeys(molecules_reference, '') # convert the zinc data set into a dictionary

        # Set up Generation Loop 
        total_time = time.time()
        for generation_index in range(1, max_generations+1):
            print("   ###   On generation %i of %i"%(generation_index, max_generations))
                  
            # Obtain molecules from the previous generation 
            smiles_here, selfies_here = gen_func.obtain_previous_gen_mol(starting_smiles,   starting_selfies, generation_size, 
                                                                         generation_index,  selfies_all,      smiles_all)

            # Calculate fitness of previous generation (shape: (generation_size, ))
            value = self.oracle(smiles_here)
            fitness_here, order, fitness_ordered, smiles_ordered, selfies_ordered = gen_func.obtain_fitness(
                                                                                        disc_enc_type,      smiles_here,   selfies_here,  
                                                                                        self.oracle,    
                                                                                        discriminator, generation_index,
                                                                                        max_molecules_len,  device,        generation_size,  
                                                                                        num_processors,     beta,            
                                                                                        image_dir,          data_dir,      max_fitness_collector, 
                                                                                        impose_time_adapted_pen)

            if self.finish:
                break 
            else:
                print("# of oracle calls", len(self.oracle))
            # Obtain molecules that need to be replaced & kept
            to_replace, to_keep = gen_func.apply_generation_cutoff(order, generation_size)
            # Obtain new generation of molecules 
            smiles_mutated, selfies_mutated = gen_func.obtain_next_gen_molecules(order,           to_replace,     to_keep, 
                                                                                 selfies_ordered, smiles_ordered, max_molecules_len)
            # Record in collective list of molecules 
            smiles_all, selfies_all, smiles_all_counter = gen_func.update_gen_res(smiles_all, smiles_mutated, 
                                                                                  selfies_all, selfies_mutated, smiles_all_counter)


            # Obtain data for training the discriminator (Note: data is shuffled)
            dataset_x, dataset_y = gen_func.obtain_discrm_data(disc_enc_type, molecules_reference, smiles_mutated, 
                                                               selfies_mutated, max_molecules_len, num_processors, generation_index)
            # Train the discriminator (on mutated molecules)
            if generation_index >= training_start_gen:
                discriminator = D.do_x_training_steps(dataset_x, dataset_y, discriminator, d_optimizer, d_loss_func , 
                                                      disc_epochs_per_generation, generation_index-1, device, data_dir)
                D.save_model(discriminator, generation_index-1, saved_models_dir) # Save the discriminator 




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracle = Oracle(name = oracle_name)
        optimizer = SELFIES_GA_optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)



if __name__ == "__main__":
    main() 

        
