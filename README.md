# adme_tox Overview

Python script that implements a random forest algorithm to predict several ADME/Tox classifications of bioactive molecules accompanied with a visualization technique called Uniform Manifold Approximation Projection (UMAP). This work is an amalgamation of a great many previous work by fellow researchers [ref] with an extension towards our own research work on predicting ion fragmentation by a mass spectrometer (MS).

The script is step-by-step implementation of our approach and is meant for sharing, studying, and critiquing by fellow researchers who are new and interested in the topic. I also include references when appropriate. The script follows the workflow below:

(image)

The ADME/Tox indications we study are drug liver injury (DILI) [ref], microsomal clearance (MC) [ref], non-toxicity [ref], and extreme toxicity [ref].

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

The folders are arranged as follows: 
- ./adme_tox/
  - rdkit_ecfp/
  - cddd/
  - adme_utils.py
  - cddd_adme.py
  - rdkit_ecfp.py
  
The main folder ./adme_tox/ is where all the files are, including main and utility files. The subseqent folders are "results" folders each corresponding to a choice of molecular encoding (i.e. rdkit + ecfp or cddd). In each results folder, there are more folders corresponding to each label of interest. For instance, in .adme_tox/rdkit_ecfp/, we have:

- ./admet_tox/
	- rdkit_ecfp/
  		- VeryToxic/
  		- NonToxic/
  		- MC/
  		- DILI/

Each of the "label" folder contains the outcome of the code. More description on this in the results folder section below. 

The main files are:

1. cddd_adme.py and rdkit_ecfp_adme.py optimize the hyperparameters of a random forest using a particular ADME/Tox training set and using machine-driven autoencoding and knowledge-driven molecular descriptors, respectively. Each file differs in that it contains different functions for the molecular encoding, but share the same workflow as shown in the image above. Noteworthy parts of the main files are explained in the next section. 

2. admet_utils.py defines the set of functions shared by the two main files. There are a lot of details here so please look into the script for more info. 

## Main Files

The cddd_adme.py and rdkit_ecfp_adme.py have different embedded functions which differently compute the molecular encoding, but share the same main structure. Here, we will summarize the main structure. It starts with initial parameters, such as choice of label and molecular encoding:

```ruby
## initial params
print('')
homedir = os.path.expanduser("~/")
workdir = homedir + 'Desktop/adme_tox/'
print('workdir: ' + workdir)
label = 'MC'
desc_choice = 'rdkit_ecfp'
savedir = workdir + desc_choice + '/' + label + '/'
print('savedir: ' + savedir)
load_enc = False # load previously saved molecular encoding?
load_mod = False # load model?
load_u = False # load UMAP points?
inc_test = True # include test sets?
```
In the .csv file, the input is a list of SMILE string associated with a binary classification of a yes (1) or a no (0). For e.g. under the "rdkit_ecfp/VeryToxic" folder, a 1 is a very toxic smile while a 0 is not. This logic applies to all other training and test sets. Within the main folder is a series of folder that associates with the choice of descriptors (i.e. rdkit, ecfp or cddd) and 

```ruby
datadir = workdir + 'adme_tox_dataset/' + label + '/'
print('datadir: ' + datadir)
train_fn = label + '_train'
train_ft = '.csv'
print('train fn: ' + train_fn + train_ft)
train_path = datadir + train_fn + train_ft
smiles_nonzero_list, smiles_zero_list, zero_id = getsmi_from_csv(train_path)
print('train nonzero count: ' + str(zero_id))

if load_enc: ## load previously saved embedding
	dfsmi_enc_train = pd.read_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_rdkit_ecfp.csv', index_col=0)
	count_train = dfsmi_enc_train.shape[0]
	x_train = dfsmi_enc_train.to_numpy()
	y_train = y = np.concatenate([np.ones(zero_id),
		np.zeros(count_train-zero_id)])
	print('train smiles encoding loaded')
else: ## generate embedding
	print(''.join('train smiles encodings being computed'))
	x_train, y_train, dfsmi_enc_train = \
			getrdkitdesc_from_smi(smiles_nonzero_list,
								smiles_zero_list,
								zero_id)
	dfsmi_enc_train.to_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_rdkit_ecfp.csv')
```

```ruby
if inc_test: ## include test sets
	print('test set IS included')
	test_fn = label + '_test'
	test_ft = '.csv'
	print('test fn: ' + test_fn + test_ft)
	test_path = datadir + '/' + test_fn + test_ft
	smiles_nonzero_list_test, smiles_zero_list_test, zero_test_id = \
		getsmi_from_csv(test_path)
	print(''.join(['test nonzero count: ', str(zero_test_id)]))
	if load_enc:
		dfsmi_enc_test = pd.read_csv(workdir + desc_choice + '/' + \
			label + '/' + label + '_test_smiles_rdkit_ecfp.csv', index_col=0)
		x_test = dfsmi_enc_test.to_numpy()
		count_test = dfsmi_enc_test.shape[0]
		y_test = y = np.concatenate([np.ones(zero_test_id),
			np.zeros(count_test-zero_test_id)])
		test_id = dfsmi_enc_test.shape[0]
		print('test smiles encoding loaded')
	else:
		print(''.join('test smiles encoding being computed'))
		x_test, y_test, dfsmi_enc_test = \
			getrdkitdesc_from_smi(smiles_nonzero_list_test,
								smiles_zero_list_test,
								zero_test_id)
		dfsmi_enc_test.to_csv(workdir + desc_choice + '/' + \
			label + '/' + label + '_test_smiles_rdkit_ecfp.csv')
		test_id = dfsmi_enc_test.shape[0]

else:
	print('test set NOT included')
```

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

## UMAP
x=np.concatenate((x_train, x_test))
y=np.concatenate((y_train, y_test))
draw_umap(x=x, y=y, n_comps=3, load_u=load_u, savedir=savedir, label=label)
```
## Results Folder
rdkit + ecfp.              |  cddd
:-------------------------:|:-------------------------:
![](/gif/MC_umap_3d_rdkit_ecfp.gif)   |  ![](/gif/MC_umap_3d_cddd.gif)
![](/images/MC_rfc_rdkit_ecfp.png)    |  ![](/images/MC_rfc_cddd.png)

