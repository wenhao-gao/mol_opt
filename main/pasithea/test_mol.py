"""
Test molecules on Pasithea. Visualize molecular transformations.
"""

import torch
import yaml
import numpy as np
import os

from torch import nn
from random import shuffle

from utilities import data_loader
from utilities import plot_utils

from utilities.mol_utils import multiple_selfies_to_hot, edit_hot, lst_of_logP, multiple_hot_to_indices
from utilities.utils import make_dir, change_str, use_gpu


class fc_model(nn.Module):

    def __init__(self, len_max_molec1Hot, num_of_neurons_layer1,
                 num_of_neurons_layer2, num_of_neurons_layer3):
        """
        Fully Connected layers for the RNN.
        """
        super(fc_model, self).__init__()

        # Reduce dimension upto second last layer of Encoder
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
    trained_data_prop = trained_data_prop.reshape(data.shape[0]).clone().detach().numpy()

    # compare ground truth data to modelled data
    plot_utils.test_model_before_dream(trained_data_prop, computed_data_prop,
                                       directory)


def dream_model(model, prop, largest_molecule_len, alphabet, upperbound,
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
    prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)

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
        gathered_indices = multiple_hot_to_indices(molecule_reshaped)
        prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)

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


def mol_transform(mols, model, prop, largest_molecule_len, alphabet,
                  upperbound_dr, lr_dream, dreaming_parameters, plot=False):
    """Dreaming procedure for a set of molecules. Plots and saves to file
    the logP and loss evolution over number of epochs if desired."""

    for i, mol in enumerate(mols):
        mol = torch.reshape(mol, (1, mol.shape[0], mol.shape[1]))
        (track_prop, track_mol,
         percent_valid_interm,
         track_loss,
         epoch_transformed) = dream_model(model = model,
                                          prop=prop,
                                          largest_molecule_len=largest_molecule_len,
                                          alphabet=alphabet,
                                          upperbound = upperbound_dr,
                                          data_train=mol,
                                          lr=lr_dream,
                                          **dreaming_parameters,
                                          display=False)


        mol1_prop = track_prop[0]
        mol2_prop = track_prop[len(track_prop)-1]
        mol1 = track_mol[0]
        mol2 = track_mol[len(track_mol)-1]
        transform = mol1+' --> '+mol2+', '+str(mol1_prop)+' --> '+str(mol2_prop)
        print('Transformation '+ str(i+1)+': '+transform)
        print(track_mol)

        if plot:
            plot_utils.plot_transform(prop, track_mol, track_prop,
                                      epoch_transformed, track_loss)


if __name__ == '__main__':
    # import hyperparameter and training settings from yaml
    print('Start reading data file...')
    settings=yaml.load(open("settings.yml","r"))
    test = settings['test_model']
    plot = settings['plot_transform']
    mols = settings['mols']
    file_name = settings['data_preprocess']['smiles_file']
    lr_train=settings['lr_train']
    lr_train=float(lr_train)
    lr_dream=settings['lr_dream']
    lr_dream=float(lr_dream)
    batch_size=settings['training']['batch_size']
    num_epochs = settings['training']['num_epochs']
    model_parameters = settings['model']
    dreaming_parameters = settings['dreaming']

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

    directory = change_str('dream_results/{}_{}/{}/{}' \
                           .format(data_parameters_str,
                                   training_parameters_str,
                                   upperbound_tr,
                                   lr_train))
    make_dir(directory)

    args = use_gpu()

    # data-preprocessing
    data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len = \
        data_loader.preprocess(num_mol, file_name)

    if test:
        data_train, data_test, prop_vals_train, prop_vals_test \
            = data_loader.split_train_test(data, prop_vals, num_train, 0.85)

        # also need to test if the model is fine
        model = train(directory, args, model_parameters, len_max_molec1Hot,
                      upperbound_tr, data_train, prop_vals_train, data_test,
                      prop_vals_test, lr_train, num_epochs, batch_size)
    else:
        model = load_model(directory+'/model.pt', args, len_max_molec1Hot,
                           model_parameters)

    # convert from SMILES to SELFIES
    selfies_lst, _ = data_loader.get_selfie_and_smiles_encodings(mols)

    # convert from SELFIES to one-hot encoding
    mols = multiple_selfies_to_hot(selfies_lst,
                                   largest_molecule_len,
                                   alphabet)
    mols = torch.tensor(mols, dtype=torch.float, device=args.device)

    # molecular transformations
    mol_transform(mols, model, prop, largest_molecule_len, alphabet,
                  upperbound_dr, lr_dream, dreaming_parameters, plot)
