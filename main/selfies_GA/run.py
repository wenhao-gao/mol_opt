import os
from selfies import decoder 
import time
import multiprocessing
import torch
from tensorboardX import SummaryWriter
from selfies import encoder
import discriminator as D
import evolution_functions as evo
import generation_props as gen_func



def initiate_ga(num_generations,            generation_size,    starting_selfies,max_molecules_len,
                disc_epochs_per_generation, disc_enc_type,      disc_layers,     training_start_gen,           
                device,                     properties_calc_ls, num_processors,  beta, max_fitness_collector, impose_time_adapted_pen):
    
       
    
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
    for generation_index in range(1, num_generations+1):
        print("   ###   On generation %i of %i"%(generation_index, num_generations))
        start_time = time.time()

              
        # Obtain molecules from the previous generation 
        smiles_here, selfies_here = gen_func.obtain_previous_gen_mol(starting_smiles,   starting_selfies, generation_size, 
                                                                     generation_index,  selfies_all,      smiles_all)

        # Calculate fitness of previous generation (shape: (generation_size, ))
        fitness_here, order, fitness_ordered, smiles_ordered, selfies_ordered = gen_func.obtain_fitness(disc_enc_type,      smiles_here,   selfies_here, 
                                                                                                        properties_calc_ls, discriminator, generation_index,
                                                                                                        max_molecules_len,  device,        generation_size,  
                                                                                                        num_processors,     writer,        beta,            
                                                                                                        image_dir,          data_dir,      max_fitness_collector, impose_time_adapted_pen)

        # Obtain molecules that need to be replaced & kept
        to_replace, to_keep = gen_func.apply_generation_cutoff(order, generation_size)
        
        # Obtain new generation of molecules 
        smiles_mutated, selfies_mutated = gen_func.obtain_next_gen_molecules(order,           to_replace,     to_keep, 
                                                                             selfies_ordered, smiles_ordered, max_molecules_len)
        # Record in collective list of molecules 
        smiles_all, selfies_all, smiles_all_counter = gen_func.update_gen_res(smiles_all, smiles_mutated, selfies_all, selfies_mutated, smiles_all_counter)

        
        # Obtain data for training the discriminator (Note: data is shuffled)
        dataset_x, dataset_y = gen_func.obtain_discrm_data(disc_enc_type, molecules_reference, smiles_mutated, selfies_mutated, max_molecules_len, num_processors, generation_index)
        # Train the discriminator (on mutated molecules)
        if generation_index >= training_start_gen:
            discriminator = D.do_x_training_steps(dataset_x, dataset_y, discriminator, d_optimizer, d_loss_func , disc_epochs_per_generation, generation_index-1, device, writer, data_dir)
            D.save_model(discriminator, generation_index-1, saved_models_dir) # Save the discriminator 

        print('Generation time: ', round((time.time()-start_time), 2), ' seconds')

    print('Total time: ', round((time.time()-total_time)/60, 2), ' mins')
    print('Total number of unique molecules: ', len(smiles_all_counter))
    return smiles_all_counter



if __name__ == '__main__':
        
    beta_preference = [0] 
    num_iterations  = 1

    results_dir = evo.make_clean_results_dir()
    
    exper_time = time.time()
    for i in range(num_iterations):
        for beta in beta_preference:
            
            max_fitness_collector = []
            image_dir, saved_models_dir, data_dir = evo.make_clean_directories(beta, results_dir, i) # clear directories 
            
            # Initialize new TensorBoard writers
            torch.cuda.empty_cache()
            writer = SummaryWriter()   
    
            # Initiate the Genetic Algorithm
            smiles_all_counter = initiate_ga(    num_generations            = 1000,
                                                 generation_size            = 500,
                                                 starting_selfies           = [encoder('C')],
                                                 max_molecules_len          = 81,
                                                 disc_epochs_per_generation = 10,
                                                 disc_enc_type              = 'properties_rdkit',         # 'selfies' or 'smiles' or 'properties_rdkit'
                                                 disc_layers                = [100, 10],
                                                 training_start_gen         = 0,                          # generation index to start training discriminator
                                                 device                     = 'cpu',
                                                 properties_calc_ls         = ['logP', 'SAS', 'RingP'],   # None: No properties ; 'logP', 'SAS', 'RingP', 'QED'
                                                 num_processors             = multiprocessing.cpu_count(),
                                                 beta                       = beta,
                                                 max_fitness_collector      = max_fitness_collector,
                                                 impose_time_adapted_pen    = True
                                            )
            
    print('Total Experiment time: ', (time.time()-exper_time)/60, ' mins')

        
        


        
