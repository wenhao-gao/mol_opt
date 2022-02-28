""" Script to extract just top1 scores from results. """

from pathlib import Path
import re
import json

expt_dicts = []
res_dir = Path("./results/ai4sci/res/")
write_res_dir = Path("./results/ai4sci/res-top1/")
for method_dir in res_dir.iterdir():
    if not method_dir.is_dir():
        continue

    for task_dir in method_dir.iterdir():
        if not task_dir.is_dir():
            continue

        for res_file in task_dir.glob("*.json"):
            match = re.match(r"budget-(\d+)_Ndata-(\d+)_trial-(\d+)", res_file.stem)
            if match:
                budget = match.group(1)
                n_data = match.group(2)
                trial = match.group(3)
                with open(res_file) as f:
                    res = json.load(f)
                top_score = max(res["scores"])
                out_dir = write_res_dir / method_dir.name / task_dir.name
                out_dir.mkdir(exist_ok=True, parents=True)
                with open(out_dir / res_file.name, "w") as f:
                    json.dump(dict(top1=float(top_score)), f)
