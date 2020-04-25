from adme_utils import *

from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, Descriptors

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

## generate morgan fingerprint/ecfp
def genFP(mol, rad, nBits):
    """
		compute extended circular fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, rad, nBits=nBits)
    fp_vect = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_vect)

    return fp_vect

def compute_descriptors(mol, id_string):
    """
		compute rdkit descriptors
    """
    descriptors = [id_string]

    # Property descriptor
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.HeavyAtomMolWt(mol))
    descriptors.append(Descriptors.MolLogP(mol))
    descriptors.append(Descriptors.MolMR(mol))
    descriptors.append(Descriptors.TPSA(mol))
    # Constitutional descriptor
    descriptors.append(Descriptors.FractionCSP3(mol))
    # Atom
    descriptors.append(Descriptors.HeavyAtomCount(mol))
    descriptors.append(Descriptors.NHOHCount(mol))
    descriptors.append(Descriptors.NOCount(mol))
    descriptors.append(Descriptors.NumHAcceptors(mol))
    descriptors.append(Descriptors.NumHDonors(mol))
    descriptors.append(Descriptors.NumHeteroatoms(mol))
    #descriptors.append(Descriptors.NumBridgeheadAtoms(mol))
    #descriptors.append(Descriptors.NumSpiroAtoms(mol))
    # Bond
    descriptors.append(Descriptors.NumRotatableBonds(mol))
    # Electronic
    descriptors.append(Descriptors.NumRadicalElectrons(mol))
    descriptors.append(Descriptors.NumValenceElectrons(mol))
    descriptors.append(Descriptors.MaxPartialCharge(mol))
    descriptors.append(Descriptors.MinPartialCharge(mol))
    descriptors.append(Descriptors.MaxAbsPartialCharge(mol))
    descriptors.append(Descriptors.MinAbsPartialCharge(mol))
    # Ring
    #descriptors.append(Descriptors.NumRings(mol))
    descriptors.append(Descriptors.NumAromaticRings(mol))
    descriptors.append(Descriptors.NumSaturatedRings(mol))
    descriptors.append(Descriptors.NumAliphaticRings(mol))
    #descriptors.append(Descriptors.NumCarbocycles(mol))
    descriptors.append(Descriptors.NumAromaticCarbocycles(mol))
    descriptors.append(Descriptors.NumSaturatedCarbocycles(mol))
    descriptors.append(Descriptors.NumAliphaticCarbocycles(mol))
    #descriptors.append(Descriptors.NumHeterocycles(mol))
    descriptors.append(Descriptors.NumAromaticHeterocycles(mol))
    descriptors.append(Descriptors.NumSaturatedHeterocycles(mol))
    descriptors.append(Descriptors.NumAliphaticHeterocycles(mol))
    # Functional Groups
    descriptors.append(Descriptors.fr_Al_COO(mol))
    descriptors.append(Descriptors.fr_Al_OH(mol))
    descriptors.append(Descriptors.fr_Al_OH_noTert(mol))
    descriptors.append(Descriptors.fr_ArN(mol))
    descriptors.append(Descriptors.fr_Ar_COO(mol))
    descriptors.append(Descriptors.fr_Ar_N(mol))
    descriptors.append(Descriptors.fr_Ar_NH(mol))
    descriptors.append(Descriptors.fr_Ar_OH(mol))
    descriptors.append(Descriptors.fr_COO(mol))
    descriptors.append(Descriptors.fr_COO2(mol))
    descriptors.append(Descriptors.fr_C_O(mol))
    descriptors.append(Descriptors.fr_C_O_noCOO(mol))
    descriptors.append(Descriptors.fr_C_S(mol))
    descriptors.append(Descriptors.fr_HOCCN(mol))
    descriptors.append(Descriptors.fr_Imine(mol))
    descriptors.append(Descriptors.fr_NH0(mol))
    descriptors.append(Descriptors.fr_NH1(mol))
    descriptors.append(Descriptors.fr_NH2(mol))
    descriptors.append(Descriptors.fr_N_O(mol))
    descriptors.append(Descriptors.fr_Ndealkylation1(mol))
    descriptors.append(Descriptors.fr_Ndealkylation2(mol))
    descriptors.append(Descriptors.fr_Nhpyrrole(mol))
    descriptors.append(Descriptors.fr_SH(mol))
    descriptors.append(Descriptors.fr_aldehyde(mol))
    descriptors.append(Descriptors.fr_alkyl_carbamate(mol))
    descriptors.append(Descriptors.fr_alkyl_halide(mol))
    descriptors.append(Descriptors.fr_allylic_oxid(mol))
    descriptors.append(Descriptors.fr_amide(mol))
    descriptors.append(Descriptors.fr_amidine(mol))
    descriptors.append(Descriptors.fr_aniline(mol))
    descriptors.append(Descriptors.fr_aryl_methyl(mol))
    descriptors.append(Descriptors.fr_azide(mol))
    descriptors.append(Descriptors.fr_azo(mol))
    descriptors.append(Descriptors.fr_barbitur(mol))
    descriptors.append(Descriptors.fr_benzene(mol))
    descriptors.append(Descriptors.fr_benzodiazepine(mol))
    descriptors.append(Descriptors.fr_bicyclic(mol))
    descriptors.append(Descriptors.fr_diazo(mol))
    descriptors.append(Descriptors.fr_dihydropyridine(mol))
    descriptors.append(Descriptors.fr_epoxide(mol))
    descriptors.append(Descriptors.fr_ester(mol))
    descriptors.append(Descriptors.fr_ether(mol))
    descriptors.append(Descriptors.fr_furan(mol))
    descriptors.append(Descriptors.fr_guanido(mol))
    descriptors.append(Descriptors.fr_halogen(mol))
    descriptors.append(Descriptors.fr_hdrzine(mol))
    descriptors.append(Descriptors.fr_hdrzone(mol))
    descriptors.append(Descriptors.fr_imidazole(mol))
    descriptors.append(Descriptors.fr_imide(mol))
    descriptors.append(Descriptors.fr_isocyan(mol))
    descriptors.append(Descriptors.fr_isothiocyan(mol))
    descriptors.append(Descriptors.fr_ketone(mol))
    descriptors.append(Descriptors.fr_ketone_Topliss(mol))
    descriptors.append(Descriptors.fr_lactam(mol))
    descriptors.append(Descriptors.fr_lactone(mol))
    descriptors.append(Descriptors.fr_methoxy(mol))
    descriptors.append(Descriptors.fr_morpholine(mol))
    descriptors.append(Descriptors.fr_nitrile(mol))
    descriptors.append(Descriptors.fr_nitro(mol))
    descriptors.append(Descriptors.fr_nitro_arom(mol))
    descriptors.append(Descriptors.fr_nitro_arom_nonortho(mol))
    descriptors.append(Descriptors.fr_nitroso(mol))
    descriptors.append(Descriptors.fr_oxazole(mol))
    descriptors.append(Descriptors.fr_oxime(mol))
    descriptors.append(Descriptors.fr_para_hydroxylation(mol))
    descriptors.append(Descriptors.fr_phenol(mol))
    descriptors.append(Descriptors.fr_phenol_noOrthoHbond(mol))
    descriptors.append(Descriptors.fr_phos_acid(mol))
    descriptors.append(Descriptors.fr_phos_ester(mol))
    descriptors.append(Descriptors.fr_piperdine(mol))
    descriptors.append(Descriptors.fr_piperzine(mol))
    descriptors.append(Descriptors.fr_priamide(mol))
    descriptors.append(Descriptors.fr_prisulfonamd(mol))
    descriptors.append(Descriptors.fr_pyridine(mol))
    descriptors.append(Descriptors.fr_quatN(mol))
    descriptors.append(Descriptors.fr_sulfide(mol))
    descriptors.append(Descriptors.fr_sulfonamd(mol))
    descriptors.append(Descriptors.fr_sulfone(mol))
    descriptors.append(Descriptors.fr_term_acetylene(mol))
    descriptors.append(Descriptors.fr_tetrazole(mol))
    descriptors.append(Descriptors.fr_thiazole(mol))
    descriptors.append(Descriptors.fr_thiocyan(mol))
    descriptors.append(Descriptors.fr_thiophene(mol))
    descriptors.append(Descriptors.fr_unbrch_alkane(mol))
    descriptors.append(Descriptors.fr_urea(mol))
    # MOE-type descriptors
    descriptors.append(Descriptors.LabuteASA(mol))
    descriptors.append(Descriptors.PEOE_VSA1(mol))
    descriptors.append(Descriptors.PEOE_VSA2(mol))
    descriptors.append(Descriptors.PEOE_VSA3(mol))
    descriptors.append(Descriptors.PEOE_VSA4(mol))
    descriptors.append(Descriptors.PEOE_VSA5(mol))
    descriptors.append(Descriptors.PEOE_VSA6(mol))
    descriptors.append(Descriptors.PEOE_VSA7(mol))
    descriptors.append(Descriptors.PEOE_VSA8(mol))
    descriptors.append(Descriptors.PEOE_VSA9(mol))
    descriptors.append(Descriptors.PEOE_VSA10(mol))
    descriptors.append(Descriptors.PEOE_VSA11(mol))
    descriptors.append(Descriptors.PEOE_VSA12(mol))
    descriptors.append(Descriptors.PEOE_VSA13(mol))
    descriptors.append(Descriptors.PEOE_VSA14(mol))
    descriptors.append(Descriptors.SMR_VSA1(mol))
    descriptors.append(Descriptors.SMR_VSA2(mol))
    descriptors.append(Descriptors.SMR_VSA3(mol))
    descriptors.append(Descriptors.SMR_VSA4(mol))
    descriptors.append(Descriptors.SMR_VSA5(mol))
    descriptors.append(Descriptors.SMR_VSA6(mol))
    descriptors.append(Descriptors.SMR_VSA7(mol))
    descriptors.append(Descriptors.SMR_VSA8(mol))
    descriptors.append(Descriptors.SMR_VSA9(mol))
    descriptors.append(Descriptors.SMR_VSA10(mol))
    descriptors.append(Descriptors.SlogP_VSA1(mol))
    descriptors.append(Descriptors.SlogP_VSA2(mol))
    descriptors.append(Descriptors.SlogP_VSA3(mol))
    descriptors.append(Descriptors.SlogP_VSA4(mol))
    descriptors.append(Descriptors.SlogP_VSA5(mol))
    descriptors.append(Descriptors.SlogP_VSA6(mol))
    descriptors.append(Descriptors.SlogP_VSA7(mol))
    descriptors.append(Descriptors.SlogP_VSA8(mol))
    descriptors.append(Descriptors.SlogP_VSA9(mol))
    descriptors.append(Descriptors.SlogP_VSA10(mol))
    descriptors.append(Descriptors.SlogP_VSA11(mol))
    descriptors.append(Descriptors.SlogP_VSA12(mol))
    descriptors.append(Descriptors.EState_VSA1(mol))
    descriptors.append(Descriptors.EState_VSA2(mol))
    descriptors.append(Descriptors.EState_VSA3(mol))
    descriptors.append(Descriptors.EState_VSA4(mol))
    descriptors.append(Descriptors.EState_VSA5(mol))
    descriptors.append(Descriptors.EState_VSA6(mol))
    descriptors.append(Descriptors.EState_VSA7(mol))
    descriptors.append(Descriptors.EState_VSA8(mol))
    descriptors.append(Descriptors.EState_VSA9(mol))
    descriptors.append(Descriptors.EState_VSA10(mol))
    descriptors.append(Descriptors.EState_VSA11(mol))
    descriptors.append(Descriptors.VSA_EState1(mol))
    descriptors.append(Descriptors.VSA_EState2(mol))
    descriptors.append(Descriptors.VSA_EState3(mol))
    descriptors.append(Descriptors.VSA_EState4(mol))
    descriptors.append(Descriptors.VSA_EState5(mol))
    descriptors.append(Descriptors.VSA_EState6(mol))
    descriptors.append(Descriptors.VSA_EState7(mol))
    descriptors.append(Descriptors.VSA_EState8(mol))
    descriptors.append(Descriptors.VSA_EState9(mol))
    descriptors.append(Descriptors.VSA_EState10(mol))
    # Topological descriptors
    descriptors.append(Descriptors.BalabanJ(mol))
    descriptors.append(Descriptors.BertzCT(mol))
    descriptors.append(Descriptors.HallKierAlpha(mol))
    descriptors.append(Descriptors.Ipc(mol))
    descriptors.append(Descriptors.Kappa1(mol))
    descriptors.append(Descriptors.Kappa2(mol))
    descriptors.append(Descriptors.Kappa3(mol))
    # Connectivity descriptors
    descriptors.append(Descriptors.Chi0(mol))
    descriptors.append(Descriptors.Chi1(mol))
    descriptors.append(Descriptors.Chi0n(mol))
    descriptors.append(Descriptors.Chi1n(mol))
    descriptors.append(Descriptors.Chi2n(mol))
    descriptors.append(Descriptors.Chi3n(mol))
    descriptors.append(Descriptors.Chi4n(mol))
    descriptors.append(Descriptors.Chi0v(mol))
    descriptors.append(Descriptors.Chi1v(mol))
    descriptors.append(Descriptors.Chi2v(mol))
    descriptors.append(Descriptors.Chi3v(mol))
    descriptors.append(Descriptors.Chi4v(mol))
    # Other properties
    descriptors.append(Descriptors.qed(mol))
    # Morgan FP
    rad = 3
    nBits = 1024
    descriptors.extend(genFP(mol, rad, nBits))

    return(descriptors)

def getrdkitdesc_from_smi(smiles_nonzero_list, smiles_zero_list, zero_id):
	"""
		compute encoding of smiles
		Args:
			smiles_nonzero_list, smiles_zero_list: same as above
		Returns:
			x: smile embeddings
			y: fragment labels
	"""
	dfsmi_nonzero = pd.DataFrame({'smiles': smiles_nonzero_list})
	dfsmi_nonzero['id']=list(range(0, dfsmi_nonzero.shape[0]))
	dfsmi_zero = pd.DataFrame({'smiles': smiles_zero_list})
	dfsmi_zero['id']=list(range(0, dfsmi_zero.shape[0]))
	dfsmi_comb = pd.concat([dfsmi_nonzero, dfsmi_zero])
	# print(dfsmi_comb.head(5))
	newdf = []
	for id, row in dfsmi_comb.iterrows():

	    ## compute descriptors
	    smiles_string = dfsmi_comb['smiles'].iloc[id]
	    # print(smiles_string)
	    id_string = dfsmi_comb['id'].iloc[id]
	    mol = Chem.MolFromSmiles(smiles_string)
	    descriptors = compute_descriptors(mol, id_string)

	    # append results
	    newdf.append(descriptors)

	## convert descriptors to np array
	all_new = np.asarray(newdf)
	all_desc = all_new[:,1:].astype(float)
	all_name = all_new[:,:1]
	print('df.shape: ' + str(all_desc.shape))

	## checking rows
	nansmile = all_desc[~np.isnan(all_desc).any(axis=1)].shape
	print('df.shape w/out NaN smiles: ' + str(nansmile))

	## checking columns
	nandesc = all_desc[:,~np.any(np.isnan(all_desc), axis=0)].shape
	print('df.shape w/out NaN descriptors: ' + str(nandesc))

	## removing descriptors with NaN
	# all_desc = all_desc[:,~np.any(np.isnan(all_desc), axis=0)]

	## removing smiles with NaN descriptors
	all_desc = all_desc[~np.isnan(all_desc).any(axis=1),:]

	## minmax rescale descriptors
	scaler = MinMaxScaler(feature_range=(0, 1))
	all_desc_minmax = scaler.fit_transform(all_desc)

	## other options for scaling
	## standardize scale descriptors
	# all_desc_std = StandardScaler().fit_transform(all_desc)

	## robust scale descriptors
	# scaler = RobustScaler(quantile_range=(25, 75))
	# all_desc_robust = scaler.fit_transform(all_desc))

	dfsmi_desc_comb = pd.DataFrame(np.asarray(all_desc_minmax))
	x = dfsmi_desc_comb.to_numpy()
	count=x.shape[0]
	y = np.concatenate([np.ones(zero_id), np.zeros(count-zero_id)])

	return x, y, dfsmi_desc_comb

## initial params
print('')
homedir = os.path.expanduser("~/")
workdir = homedir + 'Desktop/adme_tox/' ## I like to put stuff on Desktop
print('workdir: ' + workdir)
label = 'MC'
desc_choice = 'rdkit_ecfp'
savedir = workdir + desc_choice + '/' + label + '/'
print('savedir: ' + savedir)
datadir = workdir + 'adme_tox_dataset/' + label + '/'
print('datadir: ' + datadir)
train_fn = label + '_train'
train_ft = '.csv'
print('train fn: ' + train_fn + train_ft)
train_path = datadir + train_fn + train_ft
load_enc = True # load previously saved molecular encoding?
load_mod = True # load model?
load_u = False # load UMAP points?
inc_test = True # include test sets?

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
n_comps=[2,3]
for n in n_comps:
	draw_umap(x=x, y=y, n_comps=n, load_u=load_u, savedir=savedir, label=label)
