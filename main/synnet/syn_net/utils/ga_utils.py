"""
Various utilities for the genetic algorithm.
"""
import numpy as np
import scipy


def crossover(parents, offspring_size, distribution='even'):
    """
    A function that samples an offspring set through a crossover from a mating
    pool.

    Args:
        parents (numpy.ndarray): An array which represents the mating pool.
        offspring_size (int): The size of offspring pool.
        distribution (str): Key word to indicate how to sample the parent vectors.
            Choose from ['even', 'linear', 'softmax_linear']; 'even' means sample
            parents with a even probability; 'linear' means sample probability is
            linear to ranking, one scored high has better probability to be
            selected; 'softmax_linear' means the sample probability is exponential
            of linear ranking, steeper than the 'linear', for exploitation stages.
            Defaults to 'even'.

    Returns:
        offspring (numpy.ndarray): An array which represents the offspring pool.
    """
    fp_length   = parents.shape[1]
    offspring   = np.zeros((offspring_size, fp_length))
    inherit_num = np.ceil(
        np.random.normal(loc=fp_length/2, scale=fp_length/10, size=(offspring_size, ))
    )

    inherit_num = np.where(
        inherit_num >= int(fp_length/5) * np.ones((offspring_size, )),
        inherit_num, int(fp_length/5) * np.ones((offspring_size, ))
    )
    inherit_num = np.where(
        int(fp_length*4/5) * np.ones((offspring_size, )) <= inherit_num,
        int(fp_length*4/5) * np.ones((offspring_size, )),
        inherit_num
    )

    for k in range(offspring_size):
        parent1_idx = list(set(np.random.choice(fp_length, size=int(inherit_num[k]), replace=False)))
        parent2_idx = list(set(range(fp_length)).difference(set(parent1_idx)))

        if distribution == 'even':
            parent_set = parents[np.random.choice(parents.shape[0],
                                                  size=2,
                                                  replace=False)]
        elif distribution == 'linear':
            p_         = np.arange(parents.shape[0])[::-1] + 10
            parent_set = parents[np.random.choice(parents.shape[0],
                                                  size=2,
                                                  replace=False,
                                                  p=p_/np.sum(p_))]
        elif distribution == 'softmax_linear':
            p_         = np.arange(parents.shape[0])[::-1] + 10
            parent_set = parents[np.random.choice(parents.shape[0],
                                                  size=2,
                                                  replace=False,
                                                  p=scipy.special.softmax(p_))]

        offspring[k, parent1_idx] = parent_set[0][parent1_idx]
        offspring[k, parent2_idx] = parent_set[1][parent2_idx]

    return offspring

def fitness_sum(element):
    """
    Test fitness function.
    """
    return np.sum(element)

def mutation(offspring_crossover, num_mut_per_ele=1, mut_probability=0.5):
    """
    A function that samples an offspring set through a crossover from a mating
    pool.

    Args:
        offspring_crossover (numpy.ndarray): An array which represents the
            offspring pool before mutation.
        num_mut_per_ele (int): Number of bits to flip per mutation.
        mut_probability (float): The probablity of a vector to mutate.

    Returns:
        offspring_crossover (numpy.ndarray): An array represents the offspring
            pool after mutation.
    """
    b_dict    = {1:0, 0:1}
    fp_length = offspring_crossover.shape[1]
    mut_proba = np.random.random(offspring_crossover.shape[0])

    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        if mut_proba[idx] <= mut_probability:
            position = np.random.choice(fp_length,
                                        size=int(num_mut_per_ele),
                                        replace=False)
            tmp      = np.array([b_dict[int(_)] for _ in offspring_crossover[idx, position]])
            offspring_crossover[idx, position] = tmp
        else:
            pass

    return offspring_crossover

if __name__ == '__main__':

    num_parents    = 10
    fp_size        = 128
    offspring_size = 30
    ngen           = 100
    population     = np.ceil(np.random.random(size=(num_parents, fp_size)) * 2 - 1)

    print(f'Starting with {num_parents} fps with {fp_size} bits')

    scores = np.array([fitness_sum(_) for _ in population])
    print(f'Initial: {scores.mean():.3f} +/- {scores.std():.3f}')
    print(f'Scores: {scores}')

    for n in range(ngen):

        offspring      = crossover(population, offspring_size)
        offspring      = mutation(offspring, num_mut_per_ele=4, mut_probability=0.5)
        new_population = np.concatenate([population, offspring], axis=0)
        new_scores     = np.array(scores.tolist() + [fitness_sum(_) for _ in offspring])
        scores         = []

        for parent_idx in range(num_parents):
            max_score_idx = np.where(new_scores == np.max(new_scores))[0][0]
            scores.append(new_scores[max_score_idx])
            population[parent_idx, :] = new_population[max_score_idx, :]
            new_scores[max_score_idx] = -999999

        scores = np.array(scores)
        print(f'Generation {ngen}: {scores.mean()} +/- {scores.std()}')
        print(f'Scores: {scores}')
