from tdc import Oracle
from tdc import Evaluator
jnk = Oracle(name = 'JNK3')
gsk = Oracle(name = 'GSK3B')
qed = Oracle(name = 'qed')
from sa import sa
def oracle(smiles):
	scores = [qed(smiles), sa(smiles), jnk(smiles), gsk(smiles)]
	return np.mean(scores)


parser.add_argument('--max_func_calls', type=int, default=14_950) # match DST eval setting, with small error margin


def score_mol(mol, score_fn, known_value_dict):
    smiles = Chem.MolToSmiles(mol)
    if smiles not in known_value_dict:
        known_value_dict[smiles] = score_fn(smiles)
    return known_value_dict[smiles]


f_cache = dict(start_known_smiles)



# Evaluate 
new_score_tuples = [(v, k) for k, v in all_func_evals.items() if k not in start_smiles]  # scores of new molecules
new_score_tuples.sort(reverse=True)
top100_mols = [(k, v) for (v, k) in new_score_tuples[:100]]
diversity = Evaluator(name = 'Diversity')
div = diversity([t[0] for t in top100_mols])
output = dict(
        top_mols=top100_mols,
        AST=np.average([t[1] for t in top100_mols]),
        diversity=div,
        all_func_evals=dict(all_func_evals),
)
with open(args.output_file, "w") as f:
    json.dump(output, f, indent=4)




