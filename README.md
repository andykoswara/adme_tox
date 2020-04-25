# adme_tox Overview

Python script that implements a random forest algorithm to predict several ADME-Tox classifications of bioactive molecules accompanied with a visualization technique called Uniform Manifold Approximation Projection (UMAP). This work is an amalgamation of a great many previous work by fellow researchers [ref] with an extension towards our own research work on predicting ion fragmentation by a mass spectrometer (MS).

The script is step-by-step implementation of our approach and is meant for sharing, studying, and critiquing by fellow researchers who are new and interested in the topic. I also include references when appropriate. The script follows the workflow below:

(image)

The ADME-Tox indications we study here are drug liver injury (DILI) [ref], microsomal clearance (MC) [ref], non-toxicity (NonToxic) [ref], and extreme toxicity (VeryToxic) [ref] - the latter of the two are as defined by the EPA. Before we proceed with the structure of the code, please ensure that you have the correct dependencies for your python environment. 

## Dependencies

```ruby
python: 3.6.9
numpy: 1.18.1
pandas: 0.25.1
sklearn: 0.22.1          
matplotlib: 2.2.3 
rdkit: 2018.09.2.0
cddd: 0.1 
tensorflow: 1.14.0
keras: 2.0.9
gpyopt: 1.2.5 
umap: 0.3.10
seaborn: 0.9.0
scipy: 1.4.1 
imageio: 2.8.0
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
  
The **main** folder *./adme_tox/* is where all both the main execution and utility files are. The main files are categorized according to a method of encoding (i.e. rdkit+ecfp or cddd). They share a lot of the same lines of code and are redundant but for clarity. The folders within the main folder are **results** folders each corresponding to a choice of molecular encoding. In each results folder, there are **label** folders each corresponding to an ADME-Tox indication of interest. For instance, in *./adme_tox/rdkit_ecfp/*, we have *./VeryToxic/*, *./NonToxic/*, *./MC/*, and *./DILI/* folders. The same applies to *./adme_tox/cddd/*. In every label folder is the output of the main files. All the training and test sets are the **dataset** folder - *./adme_tox_dataset/* - which is again categorized according to the particular label of interest. The input is stored in .csv format. They are a list of molecular SMILE string associated with a binary classification of a yes (1) or a no (0). For e.g. within *./adme_tox_dataset/VeryToxic/VeryToxic_train.csv*, a 1 is a very toxic molecule while a 0 is not. This logic applies to all other training and test sets.

## Main Files

The main files, *cddd_main.py* and *rdkit_ecfp_main.py*, optimize the hyperparameters of a random forest algorithm using an particular training set associated with an ADME-Tox label. cddd_main.py does so using machine-driven autoencoding while rdkit_ecfp_main.py with human-driven molecular descriptors. Each file differs in that it contains different functions for the molecular encoding, but share the same workflow as shown in the image above. 

We first import all the necessary libraries and pre-defined functions from *adme_utils.py*. This script is in fact the bulk of the code and defines the libraries and functions shared by the two main files. There are a lot of details here so please look into the script for more info. 

```ruby
from adme_utils import *
```

We then inialize starting parameters, including directories, labels, and load options:

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
if inc_test: ## if test set is included
	x=np.concatenate((x_train, x_test))
	y=np.concatenate((y_train, y_test))
else:
	x = x_train
	y = y_train
n_comps=[2, 3]
for n in n_comps:
	draw_umap(x=x, y=y, n_comps=n, load_u=load_u, savedir=savedir, label=label)
```

## Output

Here is an example of an example output for the script:

```
(base) Andys-MacBook-Pro:adme_tox akoswara$ conda activate cddd
(cddd) Andys-MacBook-Pro:adme_tox akoswara$ python cddd_main.py

workdir: /Users/akoswara/Desktop/adme_tox/
savedir: /Users/akoswara/Desktop/adme_tox/cddd-enc/MC/
datadir: /Users/akoswara/Desktop/adme_tox/adme_tox_dataset/MC/
train fn: MC_train.csv
df.shape: (4323, 2)
train nonzero count: 2290
train smiles encodings being computed
test set IS included
test fn: MC_test.csv
df.shape: (546, 2)
test nonzero count: 302
test smiles encoding being computed
optimizing model
cross val loss(log): 0.28983979378874725([[-1.23842694]])
cross val loss(log): 0.2908348620778979([[-1.23499966]])
cross val loss(log): 0.28993674039224404([[-1.23809252]])
cross val loss(log): 0.28940064430577916([[-1.23994324]])
...
cross val loss(log): 0.28982348054400453([[-1.23848323]])
cross val loss(log): 0.2917216836769614([[-1.23195507]])
cross val loss(log): 0.29056832432453583([[-1.23591653]])
reconstraining parameters GP_regression.rbf
reconstraining parameters GP_regression.Gaussian_noise.variance
cross val loss(log): 0.2888356029788042([[-1.2418976]])
cross val loss(log): 0.2895220092441896([[-1.23952396]])
best rfc params: [2896.   21.    8.    6.]
model saved in: /Users/akoswara/Desktop/adme_tox/cddd-enc/MC/
test set found
UMAP 2D points being computed
UMAP 3D points being computed
```

And, here is an example of the summary of the outcome of random forest prediction of MC using 2D UMAP:

|		|	rdkit + ecfp	|	cddd
|:-----:	|	:-----:		|		:-----:
|UMAP		|	![](/rdkit_ecfp-enc/MC/MC_umap_2d.png)	|	![](/cddd-enc/MC/MC_umap_2d.png)
|ROC-AUC	|	![](/rdkit_ecfp-enc/MC/MC_rfc.png)	|	![](/cddd-enc/MC/MC_rfc.png)
|accuracy	|	0.357 +/- 0.003		|	0.560 +/- 0.008
|sensitivity	|	0.996 +/- 0.011		|	0.806 +/- 0.002
|specificity	|	0.0045 +/- 0.014	|	0.680 +/- 0.021	

and that of NonToxic using 3D UMAP:

|		|	rdkit + ecfp	|	cddd
|:-----:	|	:-----:		|		:-----:
|UMAP		|	![](/rdkit_ecfp-enc/NonToxic/NonToxic_umap_3d.gif)	|![](/cddd-enc/NonToxic/NonToxic_umap_3d.gif)
|ROC-AUC	|	![](/rdkit_ecfp-enc/NonToxic/NonToxic_rfc.png)	|	![](/cddd-enc/NonToxic/NonToxic_rfc.png)
|accuracy	|			|	0.696 +/- 0.004	
|sensitivity	|			|	0.614 +/- 0.012
|specificity	|			|	0.878 +/- 0.007

