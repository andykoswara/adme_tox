# adme_tox Overview

Python script that implements a random forest algorithm to predict several ADME/Tox classifications of bioactive molecules accompanied with a visualization technique called Uniform Manifold Approximation Projection (UMAP). This work is an amalgamation of a great many previous work by fellow researchers [ref] with an extension towards our own research work on predicting ion fragmentation by a mass spectrometer (MS).

The script is step-by-step implementation of our approach and is meant for sharing, studying, and critiquing by fellow researchers who are new and interested in the topic. I also include references when appropriate. The script follows the workflow below:

(image)

The ADME/Tox indications we study here are drug liver injury (DILI) [ref], microsomal clearance (MC) [ref], non-toxicity [ref], and extreme toxicity [ref] - the latter of the two are as defined by the EPA. Before we proceed with the structure of the code, please ensure that you have the correct dependencies for your python environment. 

## Dependencies

```ruby
python: 3.6.9
numpy: 1.18.1
pandas: 0.25.1
sklearn: 0.22.1          
scipy: 1.4.1 
matplotlib: 2.2.3 
rdkit: 2018.09.2.0
tensorflow: 1.14.0
keras: 2.0.9
gpyopt: 1.2.5          
seaborn: 0.9.0
imageio: 2.8.0
umap: 0.3.10
```
## Folder Taxonomy

The directory is arranged as follows: 
- ./adme_tox/
  - rdkit_ecfp/
  	- VeryToxic/
		- forgif/
		- NonToxic_umap_2d.png
		- NonToxic_raw_umap_2d.csv
		- NonToxic_umap_3d.gif
		- NonToxic_raw_umap_3d.csv
		- NonToxic_rfc.txt
		- NonToxic_rfc.png
		- e3606_d2_fsqrt_ms4_ml9_rfc.pkl
		- NonToxic_test_smiles_rdkit_ecfp.csv
		- NonToxic_train_smiles_rdkit_ecfp.csv
  	- NonToxic/
  	- MC/
  	- DILI/
  - cddd/
  - adme_tox_dataset/
  - adme_utils.py
  - cddd_main.py
  - rdkit_ecfp_main.py
  
The main folder ./adme_tox/ is where all both the main execution and utility files are. The main files are categorized according to a method of encoding (i.e. rdkit + ecfp or cddd). They share a lot of the same lines of code and are redundant but for clarity. The folders within the main folder are "results" folders each corresponding to a choice of molecular encoding. In each results folder, there are "label" folders each corresponding to an ADME/Tox indication of interest. For instance, in .adme_tox/rdkit_ecfp/, we have VeryToxic, Nontoxic, MC, and DILI folders. The same applies to .adme_tox/cddd/. In every label folder is the output of the main files. All the training and test sets are in adme_tox_dataset/ which is again categorized according to the particular label of interest. The input is stored in .csv format. They are a list of molecular SMILE string associated with a binary classification of a yes (1) or a no (0). For e.g. under the "adme_tox_dataset/VeryToxic/VeryToxic_train.csv" folder, a 1 is a very toxic molecule while a 0 is not. This logic applies to all other training and test sets.

## Main Files

The main files, cddd_main.py and rdkit_ecfp_main.py, optimize the hyperparameters of a random forest algorithm using an particular training set associated with an ADME/Tox label. cddd_main.py does so using machine-driven autoencoding while rdkit_ecfp_main.py with human-driven molecular descriptors. Each file differs in that it contains different functions for the molecular encoding, but share the same workflow as shown in the image above. 

We first import all the necessary libraries and pre-defined functions from adme_utils.py. This script is in fact the bulk of the code and defines the libraries and functions shared by the two main files. There are a lot of details here so please look into the script for more info. 

```ruby
from adme_utils import *
```

We then inialize starting parameters, including directories, labels, and load options. 

```ruby
## initial params
print('')
homedir = os.path.expanduser("~/")
workdir = homedir + 'Desktop/adme_tox/' ## I like to put stuff on Desktop
print('workdir: ' + workdir)
label = 'NonToxic'
desc_choice = 'rdkit_ecfp'
savedir = workdir + desc_choice + '/' + label + '/'
print('savedir: ' + savedir)
load_enc = True # load previously saved molecular encoding?
load_mod = True # load model?
load_u = False # load UMAP points?
inc_test = True # include test sets?
datadir = workdir + 'adme_tox_dataset/' + label + '/'
print('datadir: ' + datadir)
train_fn = label + '_train'
train_ft = '.csv'
print('train fn: ' + train_fn + train_ft)
train_path = datadir + train_fn + train_ft
```

We then acquire list of smiles from the training or test sets corresponding to a label 0 or 1. The smiles are then converted to the appropriate choice of encoding: 

```ruby
smiles_nonzero_list, smiles_zero_list, zero_id = getsmi_from_csv(train_path)
print('train nonzero count: ' + str(zero_id))

if load_enc: ## load previously saved encoding
	dfsmi_enc_train = pd.read_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_rdkit_ecfp.csv', index_col=0)
	count_train = dfsmi_enc_train.shape[0]
	x_train = dfsmi_enc_train.to_numpy()
	y_train = y = np.concatenate([np.ones(zero_id),
		np.zeros(count_train-zero_id)])
	print('train smiles encoding loaded')
else: ## generate encoding
	print(''.join('train smiles encodings being computed'))
	x_train, y_train, dfsmi_enc_train = \
			getrdkitdesc_from_smi(smiles_nonzero_list,
								smiles_zero_list,
								zero_id)
	dfsmi_enc_train.to_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_rdkit_ecfp.csv')
```

Once the smiles encodings have been generated, they are used as input to the algorithm. Note that the algorithm's hyperparameters are optimized using a Bayesian approach: 

```ruby
setglobal(savedir, label, x_train, y_train) ## set global parameters
if load_mod: ## load previously constructed model
	for file in os.listdir(savedir):
		if file.endswith(".pkl"):
			model_path = os.path.join(savedir, file)
			print('model located: ' + model_path)
	rfc = load_model(model_path)
	print('model loaded')
else:
	print('optimizing model')
	discrete_domain = def_optdom()
	rfc = get_optimrfc(discrete_domain=discrete_domain)
	analyze_rfc(x_train, y_train, rfc, x_test, y_test)
```

We then utilize UMAP to visulize the topology of the classfication in 2D and 3D:

```ruby
# UMAP
if inc_test:
	x=np.concatenate((x_train, x_test))
	y=np.concatenate((y_train, y_test))
else:
	x = x_train
	y = y_train
n_comps=[2, 3]
for n in n_comps:
	draw_umap(x=x, y=y, n_comps=n, load_u=load_u, savedir=savedir, label=label)
```

## Results Folder

Here is an example of the summary of the outcome of random forest prediction of MC using 3D UMAP:

encoding                   | rdkit + ecfp              |  cddd
:-------------------------:|:-------------------------:|:-------------------------:
UMAP      |  ![](/gif/MC_umap_3d_rdkit_ecfp.gif)   |  ![](/gif/MC_umap_3d_cddd.gif)
ROC-AUC  |   ![](/images/MC_rfc_rdkit_ecfp.png)    |  ![](/images/MC_rfc_cddd.png)
accuracy |					|
sensitivity | 					|
specificity |  					|
