# Contribution Guide

Thanks for your interest in contributing to our benchmark. 

## 1) Create a directory for new method

```bash
mkdir main/XXX
```

## 2) class inherited from BaseOptimizer 


```python

class XXX_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"

    def _optimize(self, oracle, config):
        ...

        population_size = config["population_size"]

        for it in range(iterations):

        	...

	        ### automatically save results 
    	    population_scores = self.score_mol(oracle, population_mol) 

        	### logging intermediate
        	self.log_intermediate(population_mol, population_scores)

```

## 3) main 

* set hyperparameter  
	* **running hyperparameter**: use parser argument with default value. 
	* **default model hyperparameter**: loading from hparam_default.yaml
	* tuning model hyperparameter: loading from hparam_tune.yaml 
* run optimizer  


```python
def main():

	#### set hyperparameter 
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

        oracle = Oracle(name = oracle_name)
        optimizer = GB_GA_Optimizer(args=args)

        if args.task == "simple":
            optimizer.optimize(oracle=oracle, config=config_default)
        elif args.task == "tune":
            optimizer.hparam_tune(oracle=oracle, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)


if __name__ == "__main__":
    main()
```

