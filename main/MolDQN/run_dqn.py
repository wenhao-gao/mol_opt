import numpy as np 
from tdc import Oracle, Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
    scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
    return np.mean(scores)

max_oracle_call = 50 
# f_cache = dict() 

def score_mol(smiles, score_fn, known_value_dict):
    if smiles not in known_value_dict:
        known_value_dict[smiles] = score_fn(smiles)
    return known_value_dict[smiles]

from utils.parsing import parse_args
# from agents.generator import DQNDirectedGenerator
from agents.agent import DQN

def main():
    args = parse_args()
    agent = DQN(
        score_fn=oracle,
        max_oracle_call = max_oracle_call, 
        q_fn = args.q_function, 
        args=args,
        param=args.parameters,
    )
    f_cache = agent.train()

    # Evaluate 
    new_score_tuples = [(v, k) for k, v in f_cache.items()]  # scores of new molecules
    new_score_tuples.sort(reverse=True)
    top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
    diversity = Evaluator(name = 'Diversity')
    div = diversity([t[0] for t in top100_mols])
    output = dict(
            top_mols=top100_mols,
            AST=np.average([t[1] for t in top100_mols]),
            diversity=div,
            all_func_evals=dict(f_cache),
    )
    print(f_cache)
    # with open(args.output_file, "w") as f:
    #     json.dump(output, f, indent=4)



if __name__ == '__main__':
    main()





