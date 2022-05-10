"""
Utilities for visualizing training and dreaming results.
"""
import matplotlib.pyplot as plt
from utilities.utils import closefig


def running_avg_test_loss(avg_test_loss, directory):
    """Plot running average test loss"""

    plt.figure()
    plt.plot(avg_test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Running average test loss')
    name = name = directory + '/runningavg_testloss'
    plt.savefig(name)
    closefig()


def test_model_after_train(calc_train, real_vals_prop_train,
               calc_test, real_vals_prop_test,
               directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with the modelled data";
    includes both test and training data."""

    plt.figure()
    plt.scatter(calc_train,real_vals_prop_train,color='red',s=40, facecolors='none')
    plt.scatter(calc_test,real_vals_prop_test,color='blue',s=40, facecolors='none')
    plt.xlim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.ylim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    plt.title('Train set (red), test set (blue)')
    name = directory + '/test_model_after_training'
    plt.savefig(name)
    closefig()


def test_model_before_dream(trained_data_prop, computed_data_prop,
                            directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with modelled data"""

    plt.figure()
    plt.scatter(trained_data_prop, computed_data_prop)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    name = directory + '/test_model_before_dreaming'
    plt.savefig(name)
    plt.show()
    closefig()


def prediction_loss(train_loss, test_loss, directory):
    """Plot prediction loss during training of model"""

    plt.figure()
    plt.plot(train_loss, color = 'red')
    plt.plot(test_loss, color = 'blue')
    plt.title('Prediction loss: training (red), test (blue)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    name = directory + '/predictionloss_test&train'
    plt.savefig(name)
    closefig()


def dreamed_histogram(prop_lst, prop, directory, prop_name='logP'):
    """Plot distribution of property values from a given list of values
    (after transformation)"""

    plt.figure()
    plt.hist(prop_lst, density=True, bins=30)
    plt.ylabel(prop_name+' - around '+str(prop))
    name = directory + '/dreamed_histogram'
    plt.savefig(name)
    closefig()


def initial_histogram(prop_dream, directory,
                      dataset_name='QM9', prop_name='logP'):
    """Plot distribution of property values from a given list of values
    (before transformation)"""

    plt.figure()
    plt.hist(prop_dream, density=True, bins=30)
    plt.ylabel(prop_name + ' - ' + dataset_name)
    name = directory + '/QM9_histogram'
    plt.savefig(name)
    closefig()


def plot_transform(target, mol, logP, epoch, loss):
    """Combine the plots for logP transformation and loss over number of
    epochs.
    - target: the target logP to be optimized.
    - logP: the transformation of logP over number of epochs.
    - epoch: all epoch #'s where the molecule transformed when dreaming.
    - loss: loss values over number of epochs.
    """

    full_epoch = []
    full_logP = []
    step = -1
    for i in range(len(loss)):
        if i in epoch:
            step += 1
        full_logP.append(logP[step])
        full_epoch.append(i)

    fig, ax1 = plt.subplots()

    color = '#550000'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('LogP', color=color)
    ax1.plot(full_logP, linewidth=1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = '#000055'
    ax2.set_ylabel('Training loss', color=color)
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Target logP = '+str(target))
    plt.tight_layout()
    #plt.savefig('dream_results/{}_{}_transforms.svg'.format(target, loss[len(loss)-1]))

    plt.show()
