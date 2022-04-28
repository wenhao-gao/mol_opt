#!/usr/bin/env python3
"""
Example script demonstrating molecular transformations, using logP as target.
The entire cycle - training and dreaming - is involved.
"""


import os, pickle, torch, random, argparse
import yaml
import numpy as np 
from tqdm import tqdm 
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from tdc import Oracle
import sys
# sys.path.append('../..')
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
import selfies as sf 
sys.path.append('datasets')
import time

from utilities import data_loader
from utilities import plot_utils
from utilities import mol_utils

from random import shuffle
from torch import nn

from utilities.utils import change_str, make_dir, use_gpu
from utilities.mol_utils import edit_hot, lst_of_logP, multiple_hot_to_indices


class fc_model(nn.Module):

    def __init__(self, len_max_molec1Hot, num_of_neurons_layer1,
                 num_of_neurons_layer2, num_of_neurons_layer3):
        """
        Fully Connected layers for the RNN.
        """
        super(fc_model, self).__init__()

        # Reduce dimension up to second last layer of Encoder
        self.encode_4d = nn.Sequential(
            nn.Linear(len_max_molec1Hot, num_of_neurons_layer1),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer1, num_of_neurons_layer2),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer2, num_of_neurons_layer3),
            nn.ReLU(),
            nn.Linear(num_of_neurons_layer3, 1)
        )


    def forward(self, x):
        """
        Pass through the model
        """
        # Go down to dim-4
        h1 = self.encode_4d(x)

        return h1


def train_model(parent_dir, directory, args, model,
                upperbound, data_train, data_train_prop, data_test,
                data_test_prop, lr_enc, num_epochs, batch_size):
    """Train the model"""

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=lr_enc)

    # reshape for efficient parallelization
    data_train=torch.tensor(data_train, dtype=torch.float, device=args.device)
    data_test=torch.tensor(data_test, dtype=torch.float, device=args.device)
    reshaped_data_train = torch.reshape(data_train,
                                        (data_train.shape[0],
                                         data_train.shape[1]*data_train.shape[2]))
    reshaped_data_test = torch.reshape(data_test,
                                       (data_test.shape[0],
                                        data_test.shape[1]*data_test.shape[2]))

    # add random noise to one-hot encoding
    reshaped_data_test_edit = edit_hot(reshaped_data_test, upperbound)

    data_train_prop=torch.tensor(data_train_prop,
                                 dtype=torch.float, device=args.device)
    data_test_prop=torch.tensor(data_test_prop,
                                dtype=torch.float, device=args.device)

    test_loss=[]
    train_loss=[]
    avg_test_loss=[]
    min_loss = 1

    for epoch in range(num_epochs):

        # add stochasticity to the training
        x = [i for i in range(len(reshaped_data_train))]  # random shuffle input
        shuffle(x)
        reshaped_data_train  = reshaped_data_train[x]
        data_train_prop = data_train_prop[x]
        reshaped_data_train_edit = edit_hot(reshaped_data_train,
                                            upper_bound=upperbound)

        for batch_iteration in range(int(len(reshaped_data_train_edit)/batch_size)):

            current_smiles_start, current_smiles_stop = \
                batch_iteration * batch_size, (batch_iteration + 1) * batch_size

            # slice data into batches
            curr_mol=reshaped_data_train_edit[current_smiles_start : \
                                              current_smiles_stop]
            curr_prop=data_train_prop[current_smiles_start : \
                                      current_smiles_stop]

            # feedforward step
            calc_properties = model(curr_mol)
            calc_properties=torch.reshape(calc_properties,[len(calc_properties)])

            # mean-squared error between calculated property and modelled property
            criterion = nn.MSELoss()
            real_loss=criterion(calc_properties, curr_prop)

            loss = torch.clamp(real_loss, min = 0., max = 50000.).double()

            # backpropagation step
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()

        # calculate train set
        calc_train_set_property = model(reshaped_data_train_edit)
        calc_train_set_property=torch.reshape(calc_train_set_property,
                                              [len(calc_train_set_property)])
        criterion = nn.MSELoss()
        real_loss_train=criterion(calc_train_set_property, data_train_prop)
        real_loss_train_num=real_loss_train.detach().cpu().numpy()

        # calculate test set
        calc_test_set_property = model(reshaped_data_test_edit)
        criterion = nn.MSELoss()
        calc_test_set_property=torch.reshape(calc_test_set_property,
                                             [len(calc_test_set_property)])
        real_loss_test=criterion(calc_test_set_property, data_test_prop)
        real_loss_test_num=real_loss_test.detach().cpu().numpy()


        print('epoch: '+str(epoch)+' - avg loss: '+ \
              str(np.mean(real_loss_train_num))+', testset: '+ \
              str(np.mean(real_loss_test_num)))

        test_loss.append(real_loss_test_num)
        train_loss.append(real_loss_train_num)

        if real_loss_test_num < min_loss:
            min_loss = real_loss_test_num
            torch.save(model.state_dict(), parent_dir)

            print('Test loss decrease, model saved to file')

        # stopping criterion: compare the running test loss averages over 90 epochs
        if len(test_loss)>=100:
            avg = sum(test_loss[len(test_loss)-90:len(test_loss)])
            avg_test_loss.append(avg)

            print(avg_test_loss)

            if len(avg_test_loss)>=50 and avg>avg_test_loss[len(avg_test_loss)-40]:
                print('Train loss is increasing, stop training')

                # plot training results
                real_vals_prop_train=data_train_prop.detach().cpu().numpy()
                real_vals_prop_test=data_test_prop.detach().cpu().numpy()

                calc_train=calc_train_set_property.detach().cpu().numpy()
                calc_test=calc_test_set_property.detach().cpu().numpy()

                plot_utils.running_avg_test_loss(avg_test_loss, directory)
                plot_utils.test_model_after_train(calc_train, real_vals_prop_train,
                                                  calc_test,real_vals_prop_test,
                                                  directory)
                plot_utils.prediction_loss(train_loss, test_loss, directory)
                break


def load_model(file_name, args, len_max_molec1Hot, model_parameters):
    """Load existing model state dict from file"""

    model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model


def train(directory, args, model_parameters, len_max_molec1Hot, upperbound,
          data_train, prop_vals_train, data_test, prop_vals_test, lr_train,
          num_epochs, batch_size):
    name = change_str(directory)+'/model.pt'

    if os.path.exists(name):
        model = load_model(name, args, len_max_molec1Hot, model_parameters)
        print('Testing model...')
        test_model(directory, args, model,
                   data_train, prop_vals_train, upperbound)
    else:
        print('No models saved in file with current settings.')
        model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
        model.train()

        print('len(data_train): ',len(data_train))
        print("start training")

        train_model(name, directory, args, model, upperbound,
                    data_train, prop_vals_train, data_test, prop_vals_test,
                    lr_train, num_epochs, batch_size)

        model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
        model.load_state_dict(torch.load(name))
        model.eval()
        print('Testing model...')
        test_model(directory, args, model,
                   data_train, prop_vals_train, upperbound)
        print('finished training and testing, now start dreaming :)\n\n\n')

    return model


def test_model(directory, args, model, data, data_prop, upperbound):
    """Test model to ensure it is sufficiently trained before dreaming."""

    test_data = torch.tensor(data, dtype=torch.float, device=args.device)
    computed_data_prop = torch.tensor(data_prop, device=args.device)

    # reshape for efficient parallelization
    test_data = test_data.reshape(test_data.shape[0],
                                  test_data.shape[1] * test_data.shape[2])

    # add random noise to one-hot encoding with specified upperbound
    test_data_edit = edit_hot(test_data, upperbound)

    # feedforward step
    trained_data_prop = model(test_data_edit)
    trained_data_prop = trained_data_prop.reshape(data.shape[0]).cpu().clone().detach().numpy()

    # compare ground truth data to modelled data
    plot_utils.test_model_before_dream(trained_data_prop, computed_data_prop,
                                       directory)


def dream_model(oracle, model, prop, largest_molecule_len, alphabet, upperbound,
                data_train, lr, batch_size, num_epochs, display=True):
    """
    Trains in the inverse of the model with a single molecular input.
    Returns initial, final, and intermediate molecules/property values
    in the transformation;
    the percent of valid transformations;
    the list of loss terms during dreaming;
    and the list of epochs at which the molecule transformed during dreaming.
    """

    loss_prediction=[]

    # reshape for efficient parallelization
    data_train = data_train.reshape(data_train.shape[0],
                                    data_train.shape[1] * data_train.shape[2])

    # add random noise to one-hot encoding
    data_train_edit = edit_hot(data_train, upper_bound=upperbound)
    data_train_var=torch.autograd.Variable(data_train_edit, requires_grad=True)
    data_train_prop=torch.tensor([prop], dtype=torch.float)

    # convert one-hot encoding to SMILES molecule
    molecule_reshaped=torch.reshape(data_train_var,
                                    (1, largest_molecule_len,
                                     len(alphabet)))
    gathered_indices = multiple_hot_to_indices(molecule_reshaped)
    smiles_of_mol = []
    for ind in gathered_indices:
        smi = sf.decoder(mol_utils.indices_to_selfies(ind, alphabet))
        smiles_of_mol.append(smi)
    prop_of_mol = oracle(smiles_of_mol)
    print(' ----- oracle', len(oracle))
    # prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)

    #initiailize list of intermediate property values and molecules
    interm_prop = [prop_of_mol[0]]
    interm_mols = [smiles_of_mol[0]]

    epoch_transformed = [0]
    steps = 0
    valid_steps = 0

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam([data_train_var], lr=lr)

    for epoch in range(num_epochs):

        # feedforward step
        calc_properties = model(data_train_var)

        # mean squared error between target and calculated property
        calc_properties = calc_properties.reshape(batch_size)
        criterion = nn.MSELoss()
        real_loss=criterion(calc_properties, data_train_prop)
        loss = torch.clamp(real_loss, min = 0., max = 50000.).double()

        # backpropagation step
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        real_loss=loss.detach().numpy()
        loss_prediction.append(real_loss)


        if epoch%100==0:
            if display:
                print('epoch: ',epoch,', loss: ', real_loss)

        # convert one-hot encoding to SMILES molecule
        molecule_reshaped=torch.reshape(data_train_var,
                                        (1, largest_molecule_len,
                                         len(alphabet)))
        smiles_of_mol = []
        gathered_indices = multiple_hot_to_indices(molecule_reshaped)
        for ind in gathered_indices:
            smi = sf.decoder(mol_utils.indices_to_selfies(ind, alphabet))
            smiles_of_mol.append(smi)
        prop_of_mol = oracle(smiles_of_mol)
        if oracle.finish:
            break 
        # prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)

        if len(interm_prop)==0 or interm_prop[len(interm_prop)-1] != prop_of_mol[0]:

            # collect intermediate molecules
            interm_mols.append(smiles_of_mol[0])
            interm_prop.append(prop_of_mol[0])
            steps+=1
            epoch_transformed.append(epoch+1)

            if len(interm_prop)>1:

                # determine validity of transformation
                previous_prop = interm_prop[len(interm_prop)-2]
                current_prop = prop_of_mol[0]
                valid = (prop > previous_prop and current_prop > previous_prop) \
                        or (prop < previous_prop and current_prop < previous_prop)
                if valid:
                    valid_steps += 1

        if real_loss<1e-3:
            if display:
                print('Small loss, stop dreaming at epoch ', epoch)
            break

        if len(loss_prediction)>1000:
            if 0.99*loss_prediction[-900]<loss_prediction[-1]:
                if display:
                    print('Too small decrease, stop dreaming at epoch ', epoch)
                break

    percent_valid_transform = None
    if steps > 0:
        percent_valid_transform = valid_steps / steps *100

    return interm_prop, interm_mols, percent_valid_transform, loss_prediction, epoch_transformed


def dream(oracle, directory, args, largest_molecule_len, alphabet, model, train_time,
          upperbound, data_dream, prop_dream, prop,
          lr_train, lr_dream, num_train, num_dream, dreaming_parameters):
    """Dreaming procedure for a dataset of molecules. Saves the following
    results to file:
        - Summary of dreaming
        - All molecular transformations, mapping from initial to final
            molecule and property
        - Intermediate molecules for each transformation"""

    data_dream=torch.tensor(data_dream, dtype=torch.float, device=args.device)
    prop_dream = torch.tensor(prop_dream, dtype=torch.float, device=args.device)

    # plot initial distribution of property value in the dataset
    plot_utils.initial_histogram(prop_dream.numpy(), directory)
    avg1 = torch.mean(prop_dream).numpy()

    num_valid = 0
    num_unchanged = 0
    valid = False
    prop_lst = []
    interm = []
    transforms = []
    t= time.process_time()
    print('num of dream', num_dream)
    for i in tqdm(range(num_dream)):
        print('Molecule #'+str(i))

        # convert one-hot encoding to SMILES molecule
        mol = data_dream[i].clone()
        gathered_mols=[]
        _,max_index=mol.max(1)
        gathered_mols.append(max_index.data.cpu().numpy().tolist())
        smiles_of_mol = []
        for ind in gathered_mols:
            smi = sf.decoder(mol_utils.indices_to_selfies(ind, alphabet))
            smiles_of_mol.append(smi)
        prop_of_mol = oracle(smiles_of_mol)
        print('------- oracle', len(oracle))
        # prop_of_mol,smiles_of_mol=mol_utils.lst_of_logP(gathered_mols, alphabet)

        mol1 = smiles_of_mol[0]
        mol1_prop = prop_of_mol[0]
        train_mol = torch.reshape(mol, (1, mol.shape[0], mol.shape[1]))

        # feed molecule into the inverse-model
        (track_prop, track_mol,
         percent_valid_interm,
         track_loss,
         epoch_transformed) = dream_model(oracle = oracle, 
                                          model = model,
                                          prop=prop,
                                          largest_molecule_len=largest_molecule_len,
                                          alphabet=alphabet,
                                          upperbound = upperbound,
                                          data_train=train_mol,
                                          lr=lr_dream,
                                          **dreaming_parameters)

        # track and record results from dreaming
        prop_val = track_prop[len(track_prop)-1]
        mol2 = track_mol[len(track_mol)-1]
        valid = (prop > mol1_prop and prop_val > mol1_prop) or \
                (prop < mol1_prop and prop_val < mol1_prop)
        if valid:
            num_valid += 1
        if mol1_prop == prop_val or mol1==mol2:
            num_unchanged += 1
        percent_valid = num_valid*100/(i+1)
        percent_unchanged = num_unchanged*100/(i+1)
        percent_invalid = 100 - percent_valid -percent_unchanged
        transform = mol1+' --> '+mol2+', '+str(mol1_prop)+' --> '+str(prop_val)

        prop_lst.append(prop_val)
        transforms.append(transform)
        interm_tuple = ([mol1_prop]+track_prop, [mol1]+track_mol)
        interm.append(interm_tuple)

        if oracle.finish:
            break 

    # dream_time = time.process_time()-t

    # plot final distribution of property value after transformation
    # plot_utils.dreamed_histogram(prop_lst, prop, directory)

    # avg2 = sum(prop_lst)/len(prop_lst)

    # save a summary of the dreaming results to file
    # name = directory + '/summary'
    # f = open(name, "w+")

    # f.write('Summary of dreaming:\n\n')
    # f.write('Input upperbound='+str(upperbound) +'\n')
    # f.write('Target logP='+str(prop)+'\n')
    # f.write('Prediction lr='+str(lr_train)+'\n')
    # f.write('Dreaming lr='+str(lr_dream)+'\n')
    # f.write('Number of molecules trained:'+str(num_train)+'\n')
    # f.write('Number of molecules dreamed:'+str(num_dream)+'\n')
    # f.write('avg before dreaming: '+str(avg1)+'\n')
    # f.write('avg after dreaming: '+str(avg2)+'\n')
    # f.write('Percent unchanged: '+str(percent_unchanged)+'%\n')
    # f.write('Percent adjusted toward target: '+str(percent_valid)+'%\n')
    # f.write('Percent adjusted away from target: '+str(percent_invalid)+'%\n')
    # f.write('Dreaming time: '+str(dream_time)+'\n')
    # f.write('Training time: '+str(train_time)+'\n')

    # f.close()

    # save list of all transformations to file
    # name = directory +'/original_to_dream_mol'
    # g = open(name, "w+")
    # for t in transforms:
    #     g.write(t+'\n')
    # g.close()

    # # save intermediate molecules for each transformation to file
    # name = directory + '/sampled_intermediate_mol'
    # h1 = open(name, "w+")
    # for i in range(len(interm)):
    #     h1.write('Sample '+str(i+1)+'\n')
    #     h1.write(str(interm[i][0])+'\n')
    #     h1.write(str(interm[i][1])+'\n')
    # h1.close()



class Pasithea_optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "pasithea"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        settings = config 
        # import hyperparameter and training settings from yaml
        print('Start reading data file...')
        # settings=yaml.load(open(os.path.join(path_here,"settings.yml"), "r"))
        test = settings['test_model']
        plot = settings['plot_transform']
        mols = settings['mols']
        # file_name = settings['data_preprocess']['smiles_file']
        ##### write it to a new smiles_file 
        # self.all_smiles 
        file_name = str(random.randint(100000,999999))+'.txt'
        with open(file_name, 'w') as fo:
            fo.write('idx,smiles\n')
            for i,smiles in enumerate(self.all_smiles): 
                fo.write(str(i+1) + ',' + smiles + '\n')


        lr_train=settings['lr_train']
        lr_train=float(lr_train)
        lr_dream=settings['lr_dream']
        lr_dream=float(lr_dream)

        batch_size=settings['training']['batch_size']
        num_epochs = settings['training']['num_epochs']
        model_parameters = settings['model']
        dreaming_parameters = settings['dreaming']
        dreaming_parameters_str = '{}_{}'.format(dreaming_parameters['batch_size'],
                                                 dreaming_parameters['num_epochs'])
        training_parameters = settings['training']
        training_parameters_str = '{}_{}'.format(training_parameters['num_epochs'],
                                                 training_parameters['batch_size'])
        data_parameters = settings['data']
        data_parameters_str = '{}_{}'.format(data_parameters['num_train'],
                                             data_parameters['num_dream'])

        upperbound_tr = settings['upperbound_tr']
        upperbound_dr = settings['upperbound_dr']
        prop=settings['property_value']

        num_train = settings['data']['num_train']
        num_dream = settings['data']['num_dream']


        num_mol = num_train

        if num_dream > num_train:
            num_mol = num_dream
        import os 
        os.system('rm -rf dream_results')

        directory = change_str('dream_results/{}_{}/{}/{}'.format(data_parameters_str,
                                       training_parameters_str,
                                       upperbound_tr,
                                       lr_train))

        make_dir(directory)

        args = use_gpu()

        # data-preprocessing
        data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len = \
            data_loader.preprocess(num_mol, file_name, self.oracle)


        # add stochasticity to data
        x = [i for i in range(len(data))]  # random shuffle input
        shuffle(x)
        data = data[x]
        prop_vals=prop_vals[x]

        data_dream = data[:num_dream]
        prop_dream = prop_vals[:num_dream]

        data_train, data_test, prop_vals_train, prop_vals_test \
            = data_loader.split_train_test(data, prop_vals, num_train, 0.85)

        t=time.process_time()
        model = train(directory, args, model_parameters, len_max_molec1Hot,
                      upperbound_tr, data_train, prop_vals_train, data_test,
                      prop_vals_test, lr_train, num_epochs, batch_size)
        train_time = time.process_time()-t

        directory += change_str('/{}_{}'.format(upperbound_dr, dreaming_parameters_str))
        make_dir(directory)
        directory += change_str('/{}'.format(lr_dream))
        make_dir(directory)
        directory += change_str('/{}'.format(prop))
        make_dir(directory)

        dream(self.oracle, directory, args, largest_molecule_len, alphabet,
              model, train_time, upperbound_dr, data_dream,
              prop_dream, prop, lr_train, lr_dream, num_train,
              num_dream, dreaming_parameters)
        print('-----', len(self.oracle))





