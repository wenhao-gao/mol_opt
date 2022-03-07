import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evolution_functions as evo

class Net(torch.nn.Module):
    def __init__(self, n_feature, h_sizes, n_output):
        super(Net, self).__init__()
        
        # Layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        self.predict = torch.nn.Linear(h_sizes[-1], n_output)


    def forward(self, x):
        
        for layer in self.hidden:
            x = F.sigmoid(layer(x))
        output= F.sigmoid(self.predict(x))

        return output


def create_discriminator(one_hot_len, n_hidden, device):
    """
    Define an instance of the discriminator 
    """
    n_hidden.insert(0, one_hot_len)

    net = Net(n_feature=one_hot_len, h_sizes=n_hidden, n_output=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    loss_func = torch.nn.BCELoss()
    
    return (net, optimizer, loss_func)


def obtain_initial_discriminator(disc_enc_type, disc_layers, max_molecules_len, device):
    ''' Obtain Discriminator initializer
    
    Parameters:
    disc_enc_type        (str)  : (smile/selfie/properties_rdkit)
                                  For calculating num. of features to be shown to discrm.
    disc_layers,         (list) : Intermediate discrm layers (e.g. [100, 10])
    max_molecules_len    (int)  : Len of largest molecule
    device               (str)  : Device discrm. will be initialized 
    
    Returns:
    discriminator : torch model
    d_optimizer   : Loss function optimized (Adam)
    d_loss_func   : Loss (Cross-Entropy )
    '''
    # Discriminator initialization 
    if disc_enc_type == 'smiles' or disc_enc_type == 'selfies':
        alphabet = evo.smiles_alphabet(disc_enc_type)
        one_hot_len = len(alphabet) * max_molecules_len
        discriminator, d_optimizer, d_loss_func = create_discriminator(one_hot_len, disc_layers, 0.0, device) 
        return discriminator, d_optimizer, d_loss_func
    elif disc_enc_type == 'properties_rdkit':
        discriminator, d_optimizer, d_loss_func = create_discriminator(51, disc_layers, device)  
        return discriminator, d_optimizer, d_loss_func



def do_x_training_steps(data_x, data_y, net, optimizer, loss_func, steps, graph_x_counter, device, data_dir):
    
    data_x = torch.tensor(data_x.astype(np.float32), device=device)
    data_y = torch.tensor(data_y, device=device, dtype=torch.float)
    data_y = data_y.unsqueeze(-1) # -1 adds an additional dimension to target tensor    
    net.train()
    for t in range(steps):
        predictions = net(data_x)
        loss = loss_func(predictions, data_y)
        
        # TensorBoard graphing (loss and weights histogram)
        f = open('{}/discr_loss.txt'.format(data_dir), 'a+')
        f.write(str(float(loss)) + '\n')
        f.close()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        torch.cuda.empty_cache()
    return net



def do_predictions(discriminator, data_x, device):
    discriminator = discriminator.eval()
    
    data_x = torch.tensor(data_x.astype(np.float32), device=device)
    
    outputs = discriminator(data_x)
    predictions = outputs.detach().cpu().numpy() # Return as a numpy array
    return (predictions)


def _make_dir(directory):
    os.makedirs(directory)
    
    
def save_model(model, generation_index, dir_name):
    out_dir = './{}/'.format(dir_name)
    torch.save(model, '{}/{}'.format(out_dir, generation_index))


def load_saved_model(model_number):
    model = torch.load('./saved_models/{}'.format(model_number))
    model = model.eval()
    return model 
    

def check_discr_improvement(g_data, data_y, generation_index, ga_discr_lookback, loss_func, writer, device):
    """ Algorithm written by @Mario:
        
        #     g_data     = Latest generator data
        #     old_D      = Discriminator from a previous generation index
        #     new_D      = Discriminator from current generation index
        #     loss_1     = Loss of Old_D on G_data
        #     loss_2     = Loss of New_D on G_data
        #     if loss_2 < loss_1:
        #             The discriminator has improved with time
        
    @type data_y: Labels for g_data
    """
    old_D = load_saved_model(generation_index-ga_discr_lookback) # start with 0, 1, 2, ...
    new_D = load_saved_model(generation_index)   # start with 5, 6, 7, ...

    # Set models to evaluation mode
    old_D = old_D.eval()
    new_D = new_D.eval()
    g_data = torch.tensor(g_data.astype(np.float32), device=device)
    data_y = torch.tensor(data_y, device=device, dtype=torch.float)

    # Calculate loss of old descriminator 
    outputs = old_D(g_data)
    loss_old = loss_func(outputs, data_y) 

    # Calculate loss of new descriminator
    outputs = new_D(g_data)
    loss_new = loss_func(outputs, data_y)
    # Note: If (loss_old - loss_new) is +ve: Discriminator improved (i.e. loss_old > loss_new )
    writer.add_scalar('check_discr_imporovement', loss_old-loss_new, generation_index)









    



