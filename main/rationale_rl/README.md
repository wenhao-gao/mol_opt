# Multi-Objective Molecule Generation using Interpretable Substructures

This is the implementation of our ICML 2020 paper: https://arxiv.org/abs/2002.03244

## Property Predictors
The property predictors for GSK3 and JNK3 are provided in `data/gsk3/gsk3.pkl` and `data/jnk3/jnk3.pkl`. For example, to predict properties of given molecules, run
```
python properties.py --prop jnk3 < data/jnk3/rationales.txt
python properties.py --prop gsk3,jnk3 < data/dual_gsk3_jnk3/rationales.txt
```

## Rationale Extraction
The rationale extraction module will produce a list of triplets `(molecule, rationale, score)`, where `molecule` is an active compound, `rationale` is a subgraph that explains the property and `score` is its predicted score. The following script uses 4 CPU cores (can be adjusted with `--ncpu` argument):
```
python mcts.py --data data/jnk3/actives.txt --prop jnk3 --ncpu 4 > jnk3_rationales.txt
python mcts.py --data data/gsk3/actives.txt --prop gsk3 --ncpu 4 > gsk3_rationales.txt
```
To construct multi-property rationales, we can merge the single-property rationales for GSK3 and JNK3:
```
python merge_rationale.py --rationale1 data/gsk3/rationales.txt --rationale2 data/jnk3/rationales.txt > gsk3_jnk3.txt
```

## Generative Model Pre-training
The molecule completion model is pre-trained on the ChEMBL dataset. To construct the training set, run
```
python preprocess.py --train data/chembl/all.txt --ncpu 4
mkdir chembl-processed
mv tensor-* chembl-processed
```
To train the molecule completion model, run
```
python gnn_train.py --train chembl-processed --save_dir ckpt/chembl-molgen
```

## Generate molecules with specific substructures
To generate molecules that contain specific substructures (e.g. benzene), first specify a rationale file named `rationale.txt`. Here is one example file with one line for a benzene.
```
c1ccc[c:1]c1
```
where atoms marked with 1 means the model should grow this fragment from these atoms. Then run
```
python decode.py --rationale rationale.txt --model ckpt/chembl-h400beta0.3/model.20 --num_decode 1000
```
This will generate 1000 molecules with at least one benzene ring.

## GSK3 + JNK3 + QED + SA Molecule Design

This task seeks to design dual inhibitors against GSK3 and JNK3 with drug-likeness and synthetic accessibility constraints. We have already computed multi-property rationales in `data/gsk3_jnk3_qed_sa/rationales.txt`. It is a subset of GSK3-JNK3 rationales with QED > 0.6 and SA < 4.0. 

### Step 1: Fine-tuning with Policy Gradient
Given a set of rationales, the model learns to complete them into full molecules. The molecule completion model has been pre-trained on ChEMBL, and it needs to be fine-tuned so that generated molecules will satisfy all the property constraints. To fine-tune the model on the GSK3 + JNK3 + QED + SA task, run
```
python finetune.py \
  --init_model ckpt/chembl-h400beta0.3/model.20 --save_dir ckpt/tmp/ \
  --rationale data/gsk3_jnk3_qed_sa/rationales.txt --num_decode 200 --prop gsk3,jnk3,qed,sa --epoch 30 --alpha 0.5
```

### Step 2: Molecule Generation
The molecule generation script will expand the extracted rationales into full molecules. The output is a list of pairs `(rationale, molecule)`, where `molecule` is the completion of `rationale`. In the following example, each rationale is completed for 100 times, with different sampled latent vectors z.
```
python decode.py --model ckpt/gsk3_jnk3_qed_sa/model.final > outputs.txt
```

### Step 3: Evaluation
You can evaluate the outputs for the four property constraint task by
```
python properties.py --prop gsk3,jnk3,qed,sa < outputs.txt | python scripts/qed_sa_dual_eval.py --ref_path data/dual_gsk3_jnk3/actives.txt
```
Here `--ref_path` contains all the reference molecules which is used for computing the novelty score. 





## environment

```bash
conda activate rationalRL 
```



## pretrain 

```bash
python pretrain.py
```

output is `./ckpt/model`, the size is 14 M.  


## finetune & evaluation 

```bash
python finetune.py   --init_model ckpt/chembl-h400beta0.3/model.20 --save_dir ckpt/tmp/   --rationale data/gsk3_jnk3_qed_sa/rationales.txt --num_decode 200 --prop gsk3,jnk3,qed,sa --epoch 30 --alpha 0.5
```




