# Download data for contrib
curl -L https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/fpscores.pkl.gz?raw=true --output ./rdkit_contrib/fpscores.pkl.gz

# Download model checkpoints and other necessary stuff for Rexgen (only three dirs):
curl -L https://github.com/connorcoley/rexgen_direct/blob/master/rexgen_direct/core_wln_global/model-300-3-direct/model.ckpt-140000.data-00000-of-00001?raw=true \
	--output ./synth/rexgen_direct/core_wln_global/model-300-3-direct/
curl -L https://github.com/connorcoley/rexgen_direct/blob/master/rexgen_direct/rank_diff_wln/model-core16-500-3-max150-direct-useScores/model.ckpt-2400000.data-00000-of-00001?raw=true \
	--output ./synth/rexgen_direct/rank_diff_wln/model-core16-500-3-max150-direct-useScores/model.ckpt-2400000.data-00000-of-00001

# Download datasets
mkdir tmp_; cd tmp_
curl -OL https://github.com/kevinid/molecule_generator/releases/download/1.0/datasets.tar.gz
tar -xvzf datasets.tar.gz
mv -v datasets/* ../datasets/
cd ../
rm -rf tmp_

# Download ZINC250k
curl https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv \
	--output ./datasets/zinc250k.csv
