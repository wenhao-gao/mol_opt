'''
Functions that are used while a Generation is being Evaluated 
'''
import os
import multiprocessing
from rdkit import Chem
import numpy as np
from random import randrange
import discriminator as D
import evolution_functions as evo
# from SAS_calculator.sascorer import calculateScore
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()

# def calc_prop_logP(unseen_smile_ls, property_name, props_collect):
#     '''Calculate logP for each molecule in unseen_smile_ls, and record results
#        in locked dictionary props_collect 
#     '''
#     for smile in unseen_smile_ls:
#         mol, smi_canon, did_convert = evo.sanitize_smiles(smile)
#         if did_convert:                                          # ensure valid smile 
#             props_collect[property_name][smile] = evo.get_logP(mol) # Add calculation
#         else:
#             raise Exception('Invalid smile encountered while atempting to calculate logP')

        
# def calc_prop_SAS(unseen_smile_ls, property_name, props_collect):
#     '''Calculate synthetic accesibility score for each molecule in unseen_smile_ls,
#        results are recorded in locked dictionary props_collect 
#     '''
#     for smile in unseen_smile_ls:
#         mol, smi_canon, did_convert = evo.sanitize_smiles(smile)
#         if did_convert:                                         # ensure valid smile 
#             props_collect[property_name][smile] = calculateScore(mol)
#         else:
#             raise Exception('Invalid smile encountered while atempting to calculate SAS')

    

# def calc_prop_RingP(unseen_smile_ls, property_name, props_collect):
#     '''Calculate Ring penalty for each molecule in unseen_smile_ls,
#        results are recorded in locked dictionary props_collect 
#     '''
#     for smi in unseen_smile_ls:
#         mol, smi_canon, did_convert = evo.sanitize_smiles(smi)
#         if did_convert:    
#             cycle_list = mol.GetRingInfo().AtomRings() 
#             if len(cycle_list) == 0:
#                 cycle_length = 0
#             else:
#                 cycle_length = max([ len(j) for j in cycle_list ])
#             if cycle_length <= 6:
#                 cycle_length = 0
#             else:
#                 cycle_length = cycle_length - 6
#             props_collect[property_name][smi] = cycle_length
#         else:
#             raise Exception('Invalid smile encountered while atempting to calculate Ring penalty')


# def create_parr_process(chunks, property_name):
#     ''' Create parallel processes for calculation of properties
#     '''
#     # Assign data to each process 
#     process_collector    = []
#     collect_dictionaries = []
        
#     for item in chunks:
#         props_collect  = manager.dict(lock=True)
#         smiles_map_    = manager.dict(lock=True)
#         props_collect[property_name] = smiles_map_
#         collect_dictionaries.append(props_collect)
        
#         if property_name == 'logP':
#             process_collector.append(multiprocessing.Process(target=calc_prop_logP, args=(item, property_name, props_collect, )))
        
#         if property_name == 'SAS': 
#            process_collector.append(multiprocessing.Process(target=calc_prop_SAS, args=(item, property_name, props_collect, )))
            
#         if property_name == 'RingP': 
#             process_collector.append(multiprocessing.Process(target=calc_prop_RingP, args=(item, property_name, props_collect, )))
    
#     for item in process_collector:
#         item.start()
    
#     for item in process_collector: # wait for all parallel processes to finish
#         item.join()   
        
#     combined_dict = {}             # collect results from multiple processess
#     for i,item in enumerate(collect_dictionaries):
#         combined_dict.update(item[property_name])

#     return combined_dict



def score_mol(smiles, score_fn, f_cache):
    # smiles = Chem.MolToSmiles(mol)
    # global f_cache 
    is_exists = True 
    if smiles not in f_cache:
        f_cache[smiles] = score_fn(smiles)
        is_exists = False 
    print("f_cache", len(f_cache), smiles, 'exists' if is_exists else 'new')
    return f_cache[smiles]

def fitness(molecules_here,    oracle, 
            discriminator,     disc_enc_type,         generation_index,
            max_molecules_len, device,                num_processors,    beta, 
            data_dir,          max_fitness_collector, impose_time_adapted_pen):
    ''' Calculate fitness fo a generation in the GA
    
    All properties are standardized based on the mean & stddev of the zinc dataset
    
    Parameters:
    molecules_here    (list)         : List of a string of molecules
    properties_calc_ls               : Type of property to be shown to the descriminator
    discriminator     (torch.Model)  : Pytorch classifier 
    disc_enc_type     (string)       : Indicated type of encoding shown to discriminator
    generation_index  (int)          : Which generation indicator
    max_molecules_len (int)          : Largest mol length
    device            (string)       : Device of discrimnator  
    num_processors    (int)          : Number of cpu processors to parallelize over
    writer            (tensorboardX writer obj) : Tensorboard graphing tool
    beta              (int)          : Discriminator fitness parameter
    data_dir          (str)          : Directory for saving data 
    max_fitness_collector (list)     : List for collecting max fitness values
    impose_time_adapted_pen (bool)   : Impose time-adaptive discriminator penalty? 
        
    Returns:
    fitness                   (np.array) : A lin comb of properties and 
                                           discriminator predictions
    discriminator_predictions (np.array) : The predictions made by the discrimantor
    
    '''
    dataset_x = evo.obtain_discr_encoding(molecules_here, disc_enc_type, max_molecules_len, num_processors, generation_index)
    if generation_index == 1: 
        discriminator_predictions = np.zeros((len(dataset_x),1))
    else:
        discriminator_predictions = D.do_predictions(discriminator, dataset_x, device)
    

    if True:         
        molecules_here_unique = list(set(molecules_here))
        # ratio            = len(molecules_here_unique) / num_processors 
        # chunks           = evo.get_chunks(molecules_here_unique, num_processors, ratio) 
        # chunks           = [item for item in chunks if len(item) >= 1]

        fitness = []
        # global f_cache 
        for smiles in molecules_here:
            value = oracle(smiles)
            fitness.append(value)
        fitness = np.array(fitness).reshape(-1,1)
        # Plot fitness without discriminator 
        # writer.add_scalar('max fitness without discr',  max(fitness),     generation_index)
        # writer.add_scalar('avg fitness without discr',  fitness.mean(),   generation_index)
        
        max_fitness_collector.append(max(fitness)[0])

        ## Impose the beta cuttoff! --------------------------
        if impose_time_adapted_pen: 
            if generation_index > 100:
                if len(set(max_fitness_collector[-5:])) == 1: # Check if there is a sagnation for 5 generations!
                    beta = 1000
                    print('Beta cutoff imposed  index: ', generation_index)
                    f = open('{}/beta_change_log.txt'.format(data_dir), 'a+')
                    f.write(str(generation_index) + '\n')
                    f.close()
        ## beta cuttoff imposed! --------------------------

        print('beta value: ', beta)
        fitness = (beta * discriminator_predictions) + fitness

    return fitness, discriminator_predictions
    # return fitness, logP_calculated, SAS_calculated, RingP_calculated, discriminator_predictions




def obtain_fitness(disc_enc_type, smiles_here, selfies_here,  
                   oracle,  
                   discriminator, generation_index, max_molecules_len, device, 
                   generation_size, num_processors, beta, image_dir,
                   data_dir, max_fitness_collector, impose_time_adapted_pen):
    ''' Obtain fitness of generation based on choices of disc_enc_type.
        Essentially just calls 'fitness'
    '''
    # ANALYSE THE GENERATION  
    # global f_cache   
    if disc_enc_type == 'smiles' or disc_enc_type == 'properties_rdkit':
        fitness_here,  discriminator_predictions = fitness(smiles_here,  
                                                           oracle,   
                                                           discriminator, 
                                                           disc_enc_type, generation_index,   
                                                           max_molecules_len, device, num_processors, 
                                                           beta, data_dir, 
                                                           max_fitness_collector, impose_time_adapted_pen) 
    elif disc_enc_type == 'selfies':
        fitness_here,  discriminator_predictions = fitness(selfies_here,  
                                                           oracle,  
                                                           discriminator, 
                                                           disc_enc_type, generation_index,   
                                                           max_molecules_len, device, num_processors, 
                                                           beta, data_dir, 
                                                           max_fitness_collector, impose_time_adapted_pen) 

    # logP_calculated, SAS_calculated, RingP_calculated,

    fitness_here = fitness_here.reshape((generation_size, ))
    order, fitness_ordered, smiles_ordered, selfies_ordered = order_based_on_fitness(fitness_here, smiles_here, selfies_here)    

    # Order molecules based on ordering of 'smiles_ordered'
    # logP_calculated  = [logP_calculated[idx] for idx in order]
    # SAS_calculated   = [SAS_calculated[idx] for idx in order]
    # RingP_calculated = [RingP_calculated[idx] for idx in order]
    # discriminator_predictions = [discriminator_predictions[idx] for idx in order]
    
    # os.makedirs('{}/{}'.format(data_dir, generation_index))
    # #  Write ordered smiles in a text file
    # f = open('{}/{}/smiles_ordered.txt'.format(data_dir, generation_index), 'a+')
    # f.writelines(["%s\n" % item  for item in smiles_ordered])
    # f.close()
    # #  Write logP of ordered smiles in a text file
    # f = open('{}/{}/logP_ordered.txt'.format(data_dir, generation_index), 'a+')
    # f.writelines(["%s\n" % item  for item in logP_calculated])
    # f.close()
    # #  Write sas of ordered smiles in a text file
    # f = open('{}/{}/sas_ordered.txt'.format(data_dir, generation_index), 'a+')
    # f.writelines(["%s\n" % item  for item in SAS_calculated])
    # f.close()
    # #  Write ringP of ordered smiles in a text file
    # f = open('{}/{}/ringP_ordered.txt'.format(data_dir, generation_index), 'a+')
    # f.writelines(["%s\n" % item  for item in RingP_calculated])
    # f.close()
    # #  Write discriminator predictions of ordered smiles in a text file
    # f = open('{}/{}/discrP_ordered.txt'.format(data_dir, generation_index), 'a+')
    # f.writelines(["%s\n" % item  for item in discriminator_predictions])
    # f.close()
    
    
    # Add the average & max discriminator score of a generation
    # writer.add_scalar('mean discriminator score', np.array(discriminator_predictions).mean(), generation_index)
    # writer.add_scalar('max discriminator score', max(discriminator_predictions), generation_index)
    # f = open('{}/avg_discr_score.txt'.format(data_dir), 'a+')
    # f.write(str(np.array(discriminator_predictions).mean()) + '\n')
    # f.close()
    # f = open('{}/max_discr_score.txt'.format(data_dir), 'a+')
    # f.write(str(max(discriminator_predictions)[0]) + '\n')
    # f.close()
    
    #print statement for the best molecule in the generation
#    print('Best best molecule in generation ', generation_index)
#    print('    smile  : ', smiles_ordered[0])
#    print('    fitness: ', fitness_ordered[0])
#    print('    logP   : ', logP_calculated[0])
#    print('    sas    : ', SAS_calculated[0])
#    print('    ringP  : ', RingP_calculated[0])
#    print('    discrm : ', discriminator_predictions[0])
    
    # f = open('{}/best_in_generations.txt'.format(data_dir), 'a+')
    # best_gen_str = 'index: {},  smile: {}, fitness: {}, logP: {}, sas: {}, ringP: {}, discrm: {}'.format(generation_index, smiles_ordered[0], fitness_ordered[0], logP_calculated[0], SAS_calculated[0], RingP_calculated[0], discriminator_predictions[0])
    # f.write(best_gen_str + '\n')
    # f.close()

    # show_generation_image(generation_index, image_dir, smiles_ordered, fitness_ordered, logP_calculated, SAS_calculated, RingP_calculated, discriminator_predictions)    
        
    return fitness_here, order, fitness_ordered, smiles_ordered, selfies_ordered


def show_generation_image(generation_index, image_dir, smiles_ordered, fitness, logP, SAS, RingCount, discr_scores):
    ''' Plot 100 molecules with the best fitness in in a generation 
        Called after at the end of each generation. Image in each generation
        is stored with name 'generation_index.png'
    
    Images are stored in diretory './images'
    '''
    if generation_index > 1:
        A = list(smiles_ordered) 
        A = A[:100]
        if len(A) < 100 : return #raise Exception('Not enough molecules provided for plotting ', len(A))
        A = [Chem.MolFromSmiles(x) for x in A]
        
        evo.create_100_mol_image(A, "./{}/{}_ga.png".format(image_dir, generation_index), fitness, logP, SAS, RingCount, discr_scores)


def obtain_previous_gen_mol(starting_smiles,  starting_selfies, generation_size,
                            generation_index, selfies_all,      smiles_all):
    '''Obtain molecules from one generation prior.
       If generation_index is 1, only the the starting molecules are returned 
       
     Parameters:
         
     Returns: 
    
    '''
    # Obtain molecules from the previous generation 
    
    if generation_index == 1:
        
        
        randomized_smiles  = []
        randomized_selfies = []
        for i in range(generation_size): # nothing to obtain from previous gen
                                         # So, choose random moleclues from the starting list 
            index = randrange(len(starting_smiles))
            randomized_smiles.append(starting_smiles[index])
            randomized_selfies.append(starting_selfies[index])

        return randomized_smiles, randomized_selfies
    else:
        return smiles_all[generation_index-2], selfies_all[generation_index-2]
    


def order_based_on_fitness(fitness_here, smiles_here, selfies_here):
    '''Order elements of a lists (args) based om Decreasing fitness 
    '''
    order = np.argsort(fitness_here)[::-1] # Decreasing order of indices, based on fitness 
    fitness_ordered = [fitness_here[idx] for idx in order]
    smiles_ordered = [smiles_here[idx] for idx in order]
    selfies_ordered = [selfies_here[idx] for idx in order]
    
    return order, fitness_ordered, smiles_ordered, selfies_ordered


def apply_generation_cutoff(order, generation_size):
    ''' Return of a list of indices of molecules that are kept (high fitness)
        and a list of indices of molecules that are replaced   (low fitness)
        
    The cut-off is imposed using a Fermi-Function
        
    Parameters:
    order (list)          : list of molecule indices arranged in Decreasing order of fitness
    generation_size (int) : number of molecules in a generation
    
    Returns:
    to_replace (list): indices of molecules that will be replaced by random mutations of 
                       molecules in list 'to_keep'
    to_keep    (list): indices of molecules that will be kept for the following generations
    '''
    # Get the probabilities that a molecule with a given fitness will be replaced
    # a fermi function is used to smoothen the transition
    positions     = np.array(range(0, len(order))) - 0.2*float(len(order))
    probabilities = 1.0 / (1.0 + np.exp(-0.02 * generation_size * positions / float(len(order))))        
    
#    import matplotlib.pyplot as plt
#    plt.plot(positions, probabilities)
#    plt.show()
    
    to_replace = [] # all molecules that are replaced 
    to_keep    = [] # all molecules that are kept 
    for idx in range(0,len(order)):
        if np.random.rand(1) < probabilities[idx]:
            to_replace.append(idx)
        else:
            to_keep.append(idx)

    return to_replace, to_keep

    
def obtain_next_gen_molecules(order,           to_replace,     to_keep, 
                              selfies_ordered, smiles_ordered, max_molecules_len):
    ''' Obtain the next generation of molecules. Bad molecules are replaced by 
    mutations of good molecules 
    
    Parameters:
    order (list)            : list of molecule indices arranged in Decreasing order of fitness
    to_replace (list)       : list of indices of molecules to be replaced by random mutations of better molecules
    to_keep (list)          : list of indices of molecules to be kept in following generation
    selfies_ordered (list)  : list of SELFIE molecules, ordered by fitness 
    smiles_ordered (list)   : list of SMILE molecules, ordered by fitness 
    max_molecules_len (int) : length of largest molecule 

    
    Returns:
    smiles_mutated (list): next generation of mutated molecules as SMILES
    selfies_mutated(list): next generation of mutated molecules as SELFIES
    '''
    smiles_mutated = []
    selfies_mutated = []
    for idx in range(0,len(order)):
        if idx in to_replace: # smiles to replace (by better molecules)
            random_index=np.random.choice(to_keep, size=1, replace=True, p=None)[0]                             # select a random molecule that survived
            grin_new, smiles_new = evo.mutations_random_grin(selfies_ordered[random_index], max_molecules_len)  # do the mutation

            # add mutated molecule to the population
            smiles_mutated.append(smiles_new)
            selfies_mutated.append(grin_new)
        else: # smiles to keep
            smiles_mutated.append(smiles_ordered[idx])
            selfies_mutated.append(selfies_ordered[idx])
    return smiles_mutated, selfies_mutated
   
    
def obtain_discrm_data(disc_enc_type, molecules_reference, smiles_mutated, selfies_mutated, max_molecules_len, num_processors, generation_index):
    '''Obtain data that will be used to train the discriminator (inputs & labels)
    '''
    if disc_enc_type == 'smiles':
        random_dataset_selection = np.random.choice(list(molecules_reference.keys()), size=len(smiles_mutated)).tolist()
        dataset_smiles = smiles_mutated + random_dataset_selection # Generation smiles + Dataset smiles 
        dataset_x = evo._to_onehot(dataset_smiles, disc_enc_type, max_molecules_len)
        dataset_y = np.array([1 if x in molecules_reference  else 0 for x in smiles_mutated] + 
                             [1 for i in range(len(dataset_smiles)-len(smiles_mutated))])


    elif disc_enc_type == 'selfies':
        random_dataset_selection = np.random.choice(list(molecules_reference.keys()), size=len(selfies_mutated)).tolist()
        dataset_smiles = selfies_mutated + random_dataset_selection
        dataset_x = evo._to_onehot(dataset_smiles, disc_enc_type, max_molecules_len)
        dataset_y = np.array([1 if x in molecules_reference  else 0 for x in selfies_mutated] + 
                             [1 for i in range(len(dataset_smiles)-len(selfies_mutated))])
        

    elif disc_enc_type == 'properties_rdkit':
        random_dataset_selection = np.random.choice(list(molecules_reference.keys()), size=len(smiles_mutated)).tolist()
        dataset_smiles = smiles_mutated + random_dataset_selection # Generation smiles + Dataset smiles 
        dataset_x = evo.obtain_discr_encoding(dataset_smiles, disc_enc_type, max_molecules_len, num_processors, generation_index)
        dataset_y = np.array([1 if x in molecules_reference  else 0 for x in smiles_mutated] + 
                             [1 for i in range(len(dataset_smiles)-len(selfies_mutated))])

    # Shuffle training data
    order_training = np.array(range(len(dataset_smiles))) #np.arange(len(dataset_smiles))
    np.random.shuffle(order_training)
    dataset_x = dataset_x[order_training]
    dataset_y = dataset_y[order_training]

    return dataset_x, dataset_y


def update_gen_res(smiles_all, smiles_mutated, selfies_all, selfies_mutated, smiles_all_counter):
    '''Collect results that will be shared with global variables outside generations
    '''
    smiles_all.append(smiles_mutated)
    selfies_all.append(selfies_mutated)
    
    for smi in smiles_mutated:
        if smi in smiles_all_counter:
            smiles_all_counter[smi] += 1
        else:
            smiles_all_counter[smi] = 1
    
    return smiles_all, selfies_all, smiles_all_counter



        
        


    
    
    
    
    
    
    
    
    