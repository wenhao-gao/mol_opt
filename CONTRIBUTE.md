# Contribution Guide

Thanks for your interest in our benchmark! This guide was made to help you develop your model that fits our benchmark quickly. If you have a other suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue.
Don't forget to give the project a star! Thanks again!

## 1) Create a directory for new method main/MODEL_NAME 

All codes for a model should be in the `main/MODEL_NAME` directory, including pretrained model. A `README.md` is prefered to describe the method.

```bash
mkdir main/MODEL_NAME
```

## 2) Make an Optimizer class for your method

One should run the `main/MODEL_NAME/run.py` to optimize a property by:

```bash
python run.py MODEL_NAME 
```

Within this `run.py` file, the core code for optimization should be implemented in an optimizer class. One should inherit from BaseOptimizer defined in `main/optimizer.py`, in which defined all necessary infrastructures for a molecular optimization run:

```python
from xxx import xxx 

class MODEL_NAME_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        ## Your model name, used for logging your results
        self.model_name = "model_name" 

    def _optimize(self, oracle, config): 
        """
        The code for a function to optimize a oracle function with hyper-parameters defined in config
        """
        
        ## This line is necessary
        self.oracle.assign_evaluator(oracle)

        ############# Your code to optimize an oracle function #############
        ############################## START ###############################

        ## Initialization
        population_size = config["population_size"]
        ...

        ## A typical iterative optimization loop
        for it in range(iterations):

        	## Search for next batch to evaluate
            population = model(old_population)
            ...

	        ## Score the smiles strings with self.oracle, with either a list of SMILES or a SMILES as input
            ## Doing so automatically:
            ##     1) scores the new input molecules and retrieves values for old ones
            ##     2) saves results to self.mol_buffer for logging and analyzing
            ##     3) logs the results to wandb with a predefined frequency
            ##     4) determins if we reached a predefined maximum number of oracle calls
    	    population_scores = self.oracle(population) 

            ## If we reached a predefined maximum number of oracle calls, break
            ## This line could be used in 
            if self.finish:
                break

            ## or one could also use self.oracle.finish to check within a user-defined function with self.oracle
            if self.oracle.finish:
                break

            ## Once you decide to early-stop, you could use self.log_intermediate(finish=True) to fake a converged 
            ## line to maximum number of oracle calls for comparison purposes
            if converge: 
                self.log_intermediate(finish=True)
                break

        ############################### END ################################

```

## 3) Copy a main function

After implementing your optimizer class, you could add the class to the `run.py` file. Note the arguments in argparse are for task-level control, i.e., what type of runs, how many independent runs, optimize which oracle functions, etc. Hyper-parameters for molecular optimization algorithms should be defined in `main/MODEL_NAME/hparams_default.yaml` and their search space for tuning should be defined in `main/MODEL_NAME/hparams_tune.yaml`. We will explain them in the next section.

```python
from main.graph_ga.run import GB_GA_Optimizer
from main.MODEL_NAME.run import MODEL_NAME_Optimizer

...

    if args.method == 'graph_ga':
        Optimizer = GB_GA_Optimizer
    elif args.method == MODEL_NAME:
        Optimizer = MODEL_NAME_Optimizer

```

## 4) Hyperparameters

We separate hyperparameters for task-level control, defined from `argparse`, and algorithm-level control, defined from `hparam_default.yaml`. There is no clear boundary for them, but we reccomend one keep all hyperparameters in the `self._optimize` function as task-level. 

* **running hyperparameter**: parser argument. 
* **default model hyperparameter**: `hparam_default.yaml`
* **tuning model hyperparameter**: `hparam_tune.yaml` 

For algorithm-level hyperparameters, we adopt the stratforward yaml file format. One should define a default set of hyper-parameters in `main/MODEL_NAME/hparam_default.yaml`:

```python
population_size: 50
offspring_size: 100
mutation_rate: 0.02
patience: 5
max_generations: 1000
```

And the search space for hyper-parameter tuning in `main/MODEL_NAME/hparam_tune.yaml`:

```python
name: graph_ga
method: random
metric:
  goal: maximize
  name: avg_top100
parameters:
  population_size:
    values: [20, 40, 50, 60, 80, 100, 150, 200]
  offspring_size:
    values: [50, 100, 200, 300]
  mutation_rate:
    distribution: uniform
    min: 0
    max: 0.1
  patience:
    value: 5
  max_generations:
    value: 1000
```

We use the [sweep](https://docs.wandb.ai/guides/sweeps) function in [wandb](https://docs.wandb.ai) for a convenient visualization. The taml file should follow the format as above. Further detail is in this [instruction](https://docs.wandb.ai/guides/sweeps/configuration).

## 5) Run with one-line bash command line

There are three types of runs defined in our code base: 
* `simple`: A single run for testing purposes for each oracle, is the defualt.
* `production`: Multiple independent runs with various random seeds for each oracle.
* `tune`: A hyper-parameter tuning over the search space defined in `main/MODEL_NAME/hparam_tune.yaml` for each oracle.

```bash
## run a single test run on qed with wandb logging online
python run.py MODEL_NAME --wandb online
## specify multiple random seeds 
python run.py MODEL_NAME --seed 0 1 2 
## run 5 runs with different random seeds on multuple oracles with wandb logging offline
python run.py MODEL_NAME --task production --n_runs 5 --oracles qed 
## run a hyper-parameter tuning starting from smiles in a smi_file, 30 runs in total without wandb logging
python run.py MODEL_NAME --task tune --n_runs 30 --smi_file XX --wandb disabled --other_args XX 
```

One can use argparse help to check the detail description of the arguments.


## 6) Logging metrics to wandb server

The default mode for wandb logging is `offline` for the speed and memory reasons. After finishing a run, one could syncronyze teh results to the server by running:

```bash
wandb sync PATH_TO/wandb/offline-run-20220406_182133-xxxxxxxx
```

To watch the results in time, one could turn the mode to `online` by flag `wandb`. To stop wandb logging, one could turn the mode to `disabled` by flag `wandb`.






