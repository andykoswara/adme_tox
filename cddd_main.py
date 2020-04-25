from adme_utils import *

from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles

def getemb_from_smi(smiles_nonzero_list, smiles_zero_list, nonzero_id):
	"""
	    compute encoding of smiles
	    Args:
			smiles_nonzero_list, smiles_zero_list: same as above
		Returns:
			x: smile embeddings
			y: fragment labels
    """
	inference_model = InferenceModel()

	smiles_emb_nonzero = inference_model.seq_to_emb(smiles_nonzero_list)
	smiles_emb_zero = inference_model.seq_to_emb(smiles_zero_list)

	dfsmi_nonzero_eb = pd.DataFrame(smiles_emb_nonzero)
	dfsmi_zero_eb = pd.DataFrame(smiles_emb_zero)
	dfsmi_eb = pd.concat([dfsmi_nonzero_eb, dfsmi_zero_eb])

	x = dfsmi_eb.to_numpy()
	count=x.shape[0]
	y = np.concatenate([np.ones(nonzero_id), np.zeros(count-nonzero_id)])

	return x, y, dfsmi_eb

## initial params
print('')
homedir = os.path.expanduser("~/")
workdir = homedir + 'Desktop/adme_tox/' ## I like to put stuff on Desktop
print('workdir: ' + workdir)
label = 'MC'
desc_choice = 'cddd-enc'
savedir = workdir + desc_choice + '/' + label + '/'
print('savedir: ' + savedir)
datadir = workdir + 'adme_tox_dataset/' + label + '/'
print('datadir: ' + datadir)
train_fn = label + '_train'
train_ft = '.csv'
print('train fn: ' + train_fn + train_ft)
train_path = datadir + train_fn + train_ft
load_enc = False # load previously saved molecular encoding?
load_mod = False # load model?
load_u = False # load UMAP points?
inc_test = True # include test sets?

smiles_nonzero_list, smiles_zero_list, zero_id = getsmi_from_csv(train_path)
print('train nonzero count: ' + str(zero_id))

if load_enc: ## load previously saved encoding
	dfsmi_enc_train = pd.read_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_cddd.csv', index_col=0)
	count_train = dfsmi_enc_train.shape[0]
	x_train = dfsmi_enc_train.to_numpy()
	y_train = y = np.concatenate([np.ones(zero_id),
		np.zeros(count_train-zero_id)])
	print('train smiles encoding loaded')
else: ## generate encoding
	print(''.join('train smiles encodings being computed'))
	x_train, y_train, dfsmi_enc_train = \
			getemb_from_smi(smiles_nonzero_list,
							smiles_zero_list,
							zero_id)
	dfsmi_enc_train.to_csv(workdir + desc_choice + '/' + \
		label + '/' + label + '_train_smiles_cddd.csv')

if inc_test: ## include test sets
	print('test set IS included')
	test_fn = label + '_test'
	test_ft = '.csv'
	print('test fn: ' + test_fn + test_ft)
	test_path = datadir + test_fn + test_ft
	smiles_nonzero_list_test, smiles_zero_list_test, zero_test_id = \
		getsmi_from_csv(test_path)
	print(''.join(['test nonzero count: ', str(zero_test_id)]))
	if load_enc:
		dfsmi_enc_test = pd.read_csv(workdir + desc_choice + '/' + \
			label + '/' + label + '_test_smiles_cddd.csv', index_col=0)
		x_test = dfsmi_enc_test.to_numpy()
		count_test = dfsmi_enc_test.shape[0]
		y_test = y = np.concatenate([np.ones(zero_test_id),
			np.zeros(count_test-zero_test_id)])
		test_id = dfsmi_enc_test.shape[0]
		print('test smiles encoding loaded')
	else:
		print(''.join('test smiles encoding being computed'))
		x_test, y_test, dfsmi_enc_test = \
			getemb_from_smi(smiles_nonzero_list_test,
							smiles_zero_list_test,
							zero_test_id)
		dfsmi_enc_test.to_csv(workdir + desc_choice + '/' + \
			label + '/' + label + '_test_smiles_cddd.csv')
		test_id = dfsmi_enc_test.shape[0]

else:
	print('test set NOT included')
	x_test = np.array([])
	y_test = np.array([])

setglobal(savedir, label, x_train, y_train) ## set global parameters
if load_mod: ## load previously trained model
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
if inc_test:
	x=np.concatenate((x_train, x_test))
	y=np.concatenate((y_train, y_test))
else:
	x = x_train
	y = y_train
n_comps=[2, 3]
for n in n_comps:
	draw_umap(x=x, y=y, n_comps=n, load_u=load_u, savedir=savedir, label=label)
