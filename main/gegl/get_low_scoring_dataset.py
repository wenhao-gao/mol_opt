import argparse
from tqdm import tqdm

from util.smiles.dataset import load_dataset
from util.chemistry.benchmarks import load_benchmark
from util.smiles.char_dict import SmilesCharDictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--top_k", type=int, default=800)
    parser.add_argument("--benchmark_id", type=int, default=28)
    parser.add_argument("--dataset_path", type=str, default="./resource/data/zinc/test.txt")
    parser.add_argument("--output_path", type=str, default="./resource/data/zinc/logp_800.txt")
    parser.add_argument("--max_smiles_length", type=int, default=80)
    args = parser.parse_args()

    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)
    dataset = load_dataset(char_dict=char_dict, smi_path=args.dataset_path)

    benchmark, scoring_num_list = load_benchmark(args.benchmark_id)

    smi2score = dict()
    for smi in tqdm(dataset):
        score = benchmark.wrapped_objective.score(smi)
        smi2score[smi] = score

    low_scoring_smis = sorted(dataset, key=lambda smi: smi2score[smi])[: args.top_k]

    for smi in low_scoring_smis:
        print(benchmark.wrapped_objective.score(smi))

    with open(args.output_path, "w") as f:
        f.write("\n".join(low_scoring_smis) + "\n")
