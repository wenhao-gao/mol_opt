# mol_opt

## install 

```bash
conda create -n molopt python=3.7
conda activate molopt 
conda install -c rdkit rdkit 
pip install torch 
pip install guacamol 
pip install PyTDC 
pip install networkx 
pip install joblib 
pip install nltk 
```

```bash
conda activate molopt 
```

## step by step  

- runable 
- make compatible (run.py with TDC type oracle)
- hyper-param tuning (on ranolazine_mpo and zaleplon_mpo, with 3 runs per h-param set and 30 loops)
- test run (5 independent runs for each oracle, 10k oracle calls for exploration and 1k for exploitation)
- code oragnization and cleaning


## Models

|                    | `runable` | `compatible` | `hparam` | `test` | `clean` |
|--------------------|-----------|--------------|----------|--------|---------|
| **Screening**      | ✅        | ✅           | ✅       |        |         |
| **Mol PAL**        | ✅        | ✅           |          |        |         |
| **Graph Ga**       | ✅        | ✅           | ✅       |        |         |
| **SMILES GA**      | ✅        | ✅           | ✅       |        |         |
| **SELFIES GA**     | ✅        | ✅           |          |        |         |
| **SELFIES GA +D**  | ✅        | ✅           |          |        |         |
| **Graph MCTS**     | ✅        | ✅           | ✅       |        |         |
| **SMILES HC**      | ✅        | ✅           |          |        |         |
| **SELFIES HC**     |           |              |          |        |         |
| **SMILES VAE BO**  | ✅        |              |          |        |         |
| **SELFIES VAE BO** |           |              |          |        |         |
| **JTVAE BO**       | ✅        |              |          |        |         |
| **BOSS (SMILES)**  | ✅        |              |          |        |         |
| **BOSS (SELFIES)** |           |              |          |        |         |
| **Graph-GA+GP-BO** | ✅        |              |          |        |         |
| **ORGAN**          |           |              |          |        |         |
| **MolGAN**         | ✅        |              |          |        |         |
| **ChemBO**         | ✅        |              |          |        |         |
| **REINVENT**       | ✅        |              |          |        |         |
| **RationaleRL**    | ✅        |              |          |        |         |
| **MolDQN**         | ✅        |  ✅          |          |        |         |
| **MIMOSA**         | ✅        | ✅           |          |        |         |
| **MARS**           | ✅        | ✅           |          |        |         |
| **DoG-Gen**        | doing     |              |          |        |         |
| **DoG-AE BO**      | doing     |              |          |        |         |
| **SynNet**         | ✅        |              |          |        |         |
| **Pasithea**       | ✅        |              |          |        |         |
| **DST**            | ✅        | ✅           |          |        |         |
| **GFlowNet**       | ✅        |  ✅          |          |        |         |
| **GFlowNet (AL)**  |           |              |          |        |         ||

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
python main/MODEL_NAME/run.py 
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

After implementing your optimizer class, you could copy a main function from other model directories and change the class name to your class: `MODEL_NAME_Optimizer`. Note the arguments in argparse are for task-level control, i.e., what type of runs, how many independent runs, optimize which oracle functions, etc. Hyper-parameters for molecular optimization algorithms should be defined in `main/MODEL_NAME/hparams_default.yaml` and their search space for tuning should be defined in `main/MODEL_NAME/hparams_tune.yaml`. We will detail them in the next section.

```python
def main():
    # 1. hyperparameter 
    parser = argparse.ArgumentParser()
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])
    args = parser.parse_args()

    path_here = os.path.dirname(os.path.realpath(__file__))

    if args.output_dir is None:
        args.output_dir = path_here
    elif not os.path.exist(args.output_dir):
        os.mkdir(args.output_dir)


    # 2. run optimizer 
    for oracle_name in args.oracles:

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(args.output_dir, args.config_default)))

        if args.task == "tune":
            try:
                config_tune = yaml.safe_load(open(args.config_tune))
            except:
                config_tune = yaml.safe_load(open(os.path.join(args.output_dir, args.config_tune)))

        ## Here we directly use TDC oracles, if one need to optimize their own, replace 
        ## oracle to your own TDC-type oracle function.
        oracle = Oracle(name = oracle_name)
        optimizer = MODEL_NAME_Optimizer(args=args) ## Typically one only need to change this line

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()
```

## 5) Hyperparameters

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

## 5) Running

Before running, please use the following command to add current path (parent of `main`) to python path:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
```

There are three types of runs defined in our code base: 
* `simple`: A single run for testing purposes for each oracle, is the defualt.
* `production`: Multiple independent runs with various random seeds for each oracle.
* `tune`: A hyper-parameter tuning over the search space defined in `main/MODEL_NAME/hparam_tune.yaml` for each oracle.

```bash
## run a single test run on qed
python main/MODEL_NAME/run.py 
## run 5 runs with differetn random seeds on multuple oracles
python main/MODEL_NAME/run.py --task production --n_runs 5 --oracles qed jnk3 drd2 
## run a hyper-parameter tuning starting from smiles in a smi_file, 30 runs in total
python main/MODEL_NAME/run.py --task tune --n_runs 30 --smi_file XX --other_args XX 
```

One can use argparse help to check the detail description of the arguments.






