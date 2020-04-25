import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import umap
from scipy import interp
import os, sys
import imageio

import matplotlib
matplotlib.use('TkAgg') # specific to Mac OS X
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from GPyOpt.methods import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def setglobal(_savedir, _label, _x_train, _y_train):
	global savedir
	global x_train
	global y_train
	global label
	savedir = _savedir
	label = _label
	x_train = _x_train
	y_train = _y_train

def getsmi_from_csv(path):
	"""
	    Retrieve list of smile strings given a choice of fragment
	    Args:
	    	path: directory to ADME spreadsheet
		Returns:
			smiles_nonzero_list: list of smiles corresponding to non-zero
			 	reading
			smiles_zero_list: corresponding to zero reading
			nonzero_id: no of smiles with non-zero label
    """
	dfsmi = pd.read_csv(path, index_col=0)
	dfsmi = dfsmi.reset_index()
	print('df.shape: ' + str(dfsmi.shape))
	label_id_nonzero = dfsmi['label'].to_numpy().nonzero()[0]
	dfsmi_nonzero = dfsmi.iloc[label_id_nonzero,:]
	nonzero_id = dfsmi_nonzero.shape[0]
	dfsmi_zero = dfsmi.drop(label_id_nonzero)
	smiles_nonzero_list = dfsmi_nonzero['smiles'].tolist()
	smiles_zero_list = dfsmi_zero['smiles'].tolist()

	return smiles_nonzero_list, smiles_zero_list, nonzero_id

def model_rfc(n_estimators, max_depth, max_features, min_samples_split,
			min_samples_leaf, bootstrap):
	"""
	    Construct random forest model (rfc) architecture
	    Args:
			RandomForestClassifier args (https://scikit-learn.org/stable/
				modules/generated/sklearn.ensemble.RandomForestClassifier.html)
		Returns:
			loss: loss function value of 1 - roc-auc
    """
	rfc = RandomForestClassifier(n_estimators=n_estimators,
								max_depth=max_depth,
								max_features=max_features,
								min_samples_split=min_samples_split,
								min_samples_leaf=min_samples_leaf,
								bootstrap=bootstrap)

	scores = cross_val_score(rfc, x_train, y_train, cv=5,
								scoring='roc_auc')
	precision_mean = scores.mean()
	loss = 1 - precision_mean

	return loss

def def_optdom():
	"""
		Construct function to optimize via BayesianOptimization
		Args:
			x: optimization parameters
		Returns:
			fs: fitness function, i.e. log of cross-validation loss
	"""
	## discrete variable must be at the end
	discrete_domain = [{'name': 'nest', 'type': 'discrete', 'domain': \
						(range(10, 4000))},
		{'name': 'mdepth', 'type': 'discrete', 'domain': (range(2, 100))},
		{'name': 'min_samples_split', 'type': 'discrete', 'domain': \
					[2**x for x in range(1,4)]},
		{'name': 'min_samples_leaf', 'type': 'discrete', 'domain': \
					(range(1, 10))}]

	return discrete_domain

def optim_rfc(x):
	"""
		Construct function to optimize via BayesianOptimization
		Args:
			x: rfc hyperparameter optimization parameters
		Returns:
			fs: fitness function (here it's log of cross-validation loss)
	"""
	x = np.atleast_2d(x) ## must take a 2D array
	fs = np.zeros((x.shape[0],1)) ## prepare return array with similar
		## return dimension

	for i in range(x.shape[0]):
	    cross_val_loss = model_rfc(n_estimators=int(x[i,0]),
									max_depth=int(x[i,1]),
									max_features='log2',
									min_samples_split=int(x[i,2]),
									min_samples_leaf=int(x[i,3]),
									bootstrap='false')
	    fs[i] = np.log(cross_val_loss)
	    print(''.join(['cross val loss(log): ', str(cross_val_loss),
			'(', str(fs), ')']))

	return fs

def get_optimrfc(discrete_domain):
	"""
	Construct function to optimize via BayesianOptimization
	Args:
		discrete_domain: hyperparameter optimization domain
	Returns:
		rfc: optimized rfc
	"""
	myBopt = BayesianOptimization(f=optim_rfc, ## function to optimize
	    domain=discrete_domain, ## box-constrains of the problem
	    initial_design_numdata = 50, ## number data initial design
	    model_type = "GP_MCMC",
	    acquisition_type = 'EI_MCMC',
	    evaluator_type = "predictive",
	    batch_size = 1,
	    num_cores = 4,
	    exact_feval = False) ## may not always give same exact results
	                            ## everytime
	myBopt.run_optimization(max_iter = 50)
	x_best = myBopt.x_opt #myBopt.X[np.argmin(myBopt.Y)]
	print('best rfc params: ' + str(x_best)) ## for debugging

	## save model
	n_estimators = int(x_best[0])
	max_features = 'sqrt'
	max_depth = int(x_best[1])
	min_samples_split = int(x_best[2])
	min_samples_leaf = int(x_best[3])

	rfc = RandomForestClassifier(n_estimators=n_estimators,
	                            max_depth=max_depth,
	                            max_features=max_features,
	                            min_samples_split=min_samples_split,
	                            min_samples_leaf=min_samples_leaf,
	                            bootstrap='false')

	rfc.fit(x_train, y_train)
	model_pkl_fn = savedir + 'e' + str(n_estimators) + '_d' + str(max_depth) +\
					'_f' + str(max_features) + '_ms'+ str(min_samples_split) +\
					'_ml' + str(min_samples_leaf) + '_rfc.pkl'

	with open(model_pkl_fn, 'wb') as file:
	    pickle.dump(rfc, file)

	print('model saved in: ' + savedir)

	return rfc

def calc_acc(x, y, model):
	"""
		Calculate accuracy, specificity, sensitivity
		Args:
			x, y, model: same as above
		Returns:
			specificity, sensitivity, accuracy
	"""
	y_pred_class = model.predict(x)
	cm = confusion_matrix(y, y_pred_class)
	specificity = cm[0,0]/(cm[0,0] + cm[0,1])
	sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])
	accuracy = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,1] + cm[1,1])

	return specificity, sensitivity, accuracy

def analyze_rfc(x_train, y_train, rfc, x_test=[], y_test=[]):
	"""
		Analyze optimized rfc
		Args:
			x_train, y_train, rfc: same as above
			x_test, y_test: test sets
		Returns:
			None
	"""
	if x_test.any() and y_test.any():
		print('test set found')
		testopt = True
	else:
		print('test set is null')
		testopt = False

	tpr_train_comb = []
	tpr_val_comb = []
	cvscores_train = []
	cvscores_val = []
	sensitivity_scores = []
	specificity_scores = []
	accuracy_scores = []

	if testopt:
		cvscores_test = []
		tpr_test_comb = []
		sensitivity_test_scores = []
		specificity_test_scores = []
		accuracy_test_scores = []

	base_fpr = np.linspace(0, 1, 101) ## for interpolation

	for i in range(10): ## 10 cross validation
		x_train_temp, x_val_temp, y_train_temp,\
		    y_val_temp = train_test_split(x_train, y_train,
										test_size=0.2,
										random_state=i)
		## train the model
		rfc.fit(x_train_temp, y_train_temp)

		## plot rocauc
		y_pred = rfc.predict_proba(x_train_temp)
		y_val_pred = rfc.predict_proba(x_val_temp)

		auc_train = roc_auc_score(y_train_temp, y_pred[:,1])
		auc_val = roc_auc_score(y_val_temp, y_val_pred[:,1])
		cvscores_train.append(auc_train)
		cvscores_val.append(auc_val)

		specificity, sensitivity, accuracy = calc_acc(x_val_temp,
			y_val_temp, rfc)
		specificity_scores.append(specificity)
		sensitivity_scores.append(sensitivity)
		accuracy_scores.append(accuracy)

		if testopt:
			y_test_pred = rfc.predict_proba(x_test)
			auc_test = roc_auc_score(y_test, y_test_pred[:,1])
			cvscores_test.append(auc_test)

			specificity_test, sensitivity_test, accuracy_test = \
				calc_acc(x_test, y_test, rfc)
			specificity_test_scores.append(specificity_test)
			sensitivity_test_scores.append(sensitivity_test)
			accuracy_test_scores.append(accuracy_test)

		fpr_train, tpr_train, _ = roc_curve(y_train_temp, y_pred[:,1])
		fpr_val, tpr_val, _ = roc_curve(y_val_temp, y_val_pred[:,1])

		tpr_val_temp = interp(base_fpr, fpr_val, tpr_val)
		tpr_val_temp[0] = 0.0
		tpr_val_comb.append(tpr_val_temp)

		tpr_train_temp = interp(base_fpr, fpr_train, tpr_train)
		tpr_train_temp[0] = 0.0
		tpr_train_comb.append(tpr_train_temp)

		plt.plot(fpr_train, tpr_train, color='b', alpha=0.1)
		plt.plot(fpr_val, tpr_val, color='g', alpha=0.1)

		if testopt:
			fpr_test, tpr_test, _  = roc_curve(y_test, y_test_pred[:,1])
			tpr_test_temp = interp(base_fpr, fpr_test, tpr_test)
			tpr_test_temp[0] = 0.0
			tpr_test_comb.append(tpr_test_temp)
			plt.plot(fpr_test, tpr_test, color='r', alpha=0.1)

	tpr_val_comb = np.array(tpr_val_comb)
	mean_tprs_val = tpr_val_comb.mean(axis=0)
	std_val = tpr_val_comb.std(axis=0)

	tpr_train_comb = np.array(tpr_train_comb)
	mean_tprs_train = tpr_train_comb.mean(axis=0)
	std_train = tpr_train_comb.std(axis=0)

	tprs_val_upper = np.minimum(mean_tprs_val + std_val, 1)
	tprs_val_lower = mean_tprs_val - std_val

	tprs_train_upper = np.minimum(mean_tprs_train + std_train, 1)
	tprs_train_lower = mean_tprs_train - std_train

	plt.plot(base_fpr, mean_tprs_val, 'g',
		label='mean val ROC (area = %0.3f)'%np.array(cvscores_val).mean())
	plt.fill_between(base_fpr, tprs_val_lower, tprs_val_upper, color='green',
	                 alpha=0.3)
	plt.plot(base_fpr, mean_tprs_train, 'b',
		label='mean train ROC (area = %0.3f)'%np.array(cvscores_train).mean())
	plt.fill_between(base_fpr, tprs_train_lower, tprs_train_upper,
		color='blue', alpha=0.3)

	if testopt:
		tpr_test_comb = np.array(tpr_test_comb)
		mean_tprs_test = tpr_test_comb.mean(axis=0)
		std_test = tpr_test_comb.std(axis=0)
		tprs_test_upper = np.minimum(mean_tprs_test + std_test, 1)
		tprs_test_lower = mean_tprs_test - std_test
		plt.plot(base_fpr, mean_tprs_test, 'r',
			label='mean test ROC (area = %0.3f)'%np.array(cvscores_test).mean())
		plt.fill_between(base_fpr, tprs_test_lower, tprs_test_upper,
			color='red', alpha=0.3)

	plt.plot([0, 1], [0, 1], color='navy', alpha=0.3, linestyle='--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.legend(loc="lower right")
	# plt.axes().set_aspect('equal', 'datalim')
	plt.savefig(savedir + label + '_rfc.png', dpi=326)

	## write data in a file.
	wtofile = open(savedir + label + '_rfc.txt', "w")
	wtofile.write(''.join(['train cv auc: ', str(cvscores_train), '\n\n']))
	wtofile.write(''.join(['val cv auc: ', str(cvscores_val), '\n\n']))
	wtofile.write(''.join(['mean train cv auc: ',
	                       str(np.array(cvscores_train).mean()), '\n']))
	wtofile.write(''.join(['stdev train cv auc: ',
	                       str(np.array(cvscores_train).std()), '\n\n']))
	wtofile.write(''.join(['mean val cv auc: ',
	                       str(np.array(cvscores_val).mean()), '\n']))
	wtofile.write(''.join(['stdev val cv auc: ',
	                       str(np.array(cvscores_val).std()), '\n\n']))
	if testopt:
		wtofile.write(''.join(['mean test cv auc: ',
		                       str(np.array(cvscores_test).mean()), '\n']))
		wtofile.write(''.join(['stdev test cv auc: ',
		                       str(np.array(cvscores_test).std()), '\n\n']))

	wtofile.write(''.join(['mean val accuracy: ',
	                       str(np.array(accuracy_scores).mean()), '\n']))
	wtofile.write(''.join(['stdev val accuracy: ',
	                       str(np.array(accuracy_scores).std()), '\n']))
	wtofile.write(''.join(['mean val sensitivity: ',
	                       str(np.array(sensitivity_scores).mean()), '\n']))
	wtofile.write(''.join(['stdev val sensitivity: ',
	                       str(np.array(sensitivity_scores).std()), '\n']))
	wtofile.write(''.join(['mean val specificity: ',
	                       str(np.array(specificity_scores).mean()), '\n']))
	wtofile.write(''.join(['stdev val specificity: ',
	                       str(np.array(specificity_scores).std()), '\n\n']))

	if testopt:
		wtofile.write(''.join(['mean test accuracy: ',
		                       str(np.array(accuracy_test_scores).mean()),
							   '\n']))
		wtofile.write(''.join(['stdev test accuracy: ',
		                       str(np.array(accuracy_test_scores).std()),
							   '\n']))
		wtofile.write(''.join(['mean test sensitivity: ',
		                       str(np.array(sensitivity_test_scores).mean()),
							   '\n']))
		wtofile.write(''.join(['stdev test sensitivity: ',
		                       str(np.array(sensitivity_test_scores).std()),
							   '\n']))
		wtofile.write(''.join(['mean test specificity: ',
		                       str(np.array(specificity_test_scores).mean()),
							   '\n']))
		wtofile.write(''.join(['stdev test specificity: ',
		                       str(np.array(specificity_test_scores).std()),
							   '\n']))
	wtofile.close()

def load_model(model_path):
	model = pickle.load(open(model_path, 'rb'))

	return model

def draw_umap(x=[], y=[], n_comps=2, load_u=False, savedir='', label=''):
	"""
	  	Construct function to optimize via BayesianOptimization
	  	Args:
	  		umap.UMAP args(https://umap-learn.readthedocs.io/en/latest/
				parameters.html)
	  	Returns:
	  		None
  	"""
	fit = umap.UMAP(n_neighbors=2000, min_dist=0.1, n_components=n_comps,
		metric='euclidean')
	fig = plt.figure(figsize=(3.5,3.5))

	if n_comps==2:
		if load_u:
			u = pd.read_csv(savedir + label + '_raw_umap_2d.csv',
				index_col=0)
			u = u.to_numpy()
			print('UMAP 2D points loaded')
		else:
			print('UMAP 2D points being computed')
			u = fit.fit_transform(x)
			u_df = pd.DataFrame(u)
			u_df.to_csv(savedir + label + '_raw_umap_2d.csv')

		ax = fig.add_subplot(111)
		scatter = ax.scatter(u[:,0], u[:,1],
		        c=y, cmap='Dark2',
		        marker='.',
		        s=12) # zeros
		plt.title(label, fontsize=18)
		# produce a legend with the unique colors from the scatter
		lg1 = ax.legend(*scatter.legend_elements(),
					bbox_to_anchor=(1.3, 1.0),
                    loc="upper right",
					title="Classes")
		ax.add_artist(lg1)
		plt.savefig(savedir + label + '_umap_2d.png',
			dpi=326,
			bbox_extra_artists=(lg1,),
            bbox_inches='tight')

	if n_comps==3:
		if load_u:
			u = pd.read_csv(savedir + label + '_raw_umap_3d.csv',
				index_col=0)
			u = u.to_numpy()
			print('UMAP 3D points loaded')
		else:
			print('UMAP 3D points being computed')
			u = fit.fit_transform(x)
			u_df = pd.DataFrame(u)
			u_df.to_csv(savedir + label + '_raw_umap_3d.csv')
		ax = Axes3D(fig)

		plt.axis('off') # remove axes for visual appeal

		# 20 plots, for 20 different angles
		for angle in range(0,360,4):
			scatter = ax.scatter(u[:,0], u[:,1], u[:,2],
			        c=y, cmap=plt.cm.RdPu,
			        marker='o',
					edgecolors='k',
					linewidth=0.2,
			        s=12) # zeros
			ax.view_init(30,angle)
			# plt.title(label, fontsize=18)
			legend1 = ax.legend(*scatter.legend_elements(),
	                    loc="upper right", title="Classes")
			ax.add_artist(legend1)
			fn = savedir + '/forgif/' + str(angle) + '.png'
			plt.savefig(fn, dpi=326)

		make_gif(savedir + '/forgif/',
			savedir + label + '_umap_3d.gif')

def make_gif(input_folder, save_filepath):
    episode_frames = []
    time_per_step = 0.25
    for root, _, files in os.walk(input_folder):
        file_paths = [os.path.join(root, file) for file in files]
        #sorted by modified time
        file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))
        episode_frames = [imageio.imread(file_path)
                          for file_path in file_paths if \
						  file_path.endswith('.png')]
    episode_frames = np.array(episode_frames)
    imageio.mimsave(save_filepath, episode_frames, duration=time_per_step)
