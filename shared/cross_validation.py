# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classical and two step cross validation algorithms. """
import argparse
import functools
import logging
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.stats import ttest_1samp
import tqdm

from shared import get_sigma

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

def reshape_results(results):
	shift_columns = [col for col in results.columns if 'shift' in col]
	shift_metrics_columns = [
		col for col in shift_columns if ('pred_loss' in col) or ('accuracy' in col) or ('auc' in col)
	]
	results = results[shift_metrics_columns]
	results = results.transpose()
	results['py1_y0_s'] = results.index.str[6:10]
	results['py1_y0_s'] = results.py1_y0_s.str.replace('_', '')
	results['py1_y0_s'] = results.py1_y0_s.astype(float)

	results_accuracy = results[(results.index.str.contains('accuracy'))]
	results_accuracy = results_accuracy.rename(columns={
		col: f'accuracy_{col}' for col in results_accuracy.columns if col != 'py1_y0_s'
	})

	results_auc = results[(results.index.str.contains('auc'))]
	results_auc = results_auc.rename(columns={
		col: f'auc_{col}' for col in results_auc.columns if col != 'py1_y0_s'
	})
	results_loss = results[(results.index.str.contains('pred_loss'))]
	results_loss = results_loss.rename(columns={
		col: f'loss_{col}' for col in results_loss.columns if col != 'py1_y0_s'
	})
	results_final = results_accuracy.merge(results_loss, on=['py1_y0_s'])
	results_final = results_final.merge(results_auc, on=['py1_y0_s'])

	print(results_final)
	return results_final

def get_optimal_model_two_step(results_file, all_model_dir, hparams, 
	weighted_xv, dataset, data_dir, pval=0.05, kfolds=3):
	"""This function does the two step cross-validation procedure outlined in the paper.
	Args: 
		all_results: a pandas dataframe that containes columns of hyperparameters, and the 
			corresponding 
		all_model_dir: a list of all the saved model directories. Code assumes that these 
			directories also include the config.pkl file (see readme)
		hparams: a list of the column names that have the hyperparams we're
			cross-validating over
		weighted_xv: if = 'weighted_bal', it will do the cross validation using 
			weighted metrics as descibed in the cross validation section of the paper. 
			if not specified, it will do the weighted scheme if the model is weighted, 
			and unweighted if the model is unweighted. 
		dataset: either waterbirds or chexpert.
		data_dir: the directory which has all the individual experiment data
		pval: the p-value above which we reject the null hypothesis that MMD = 0. Lower 
			values prioritize robustness (i.e., lower MMD) over accuracy
		kfolds: number of subgroups to divide the validation set into to estimated
			the variance of the MMD 
	"""
	# split all_models_dir 
	all_model_dir = all_model_dir.split(',')
	# split hparams 
	hparams = hparams.split(',')

	all_results = pd.read_csv(results_file)
	sigma_results = get_sigma.get_optimal_sigma(all_model_dir, kfolds=kfolds,
		weighted_xv=weighted_xv, dataset=dataset, data_dir=data_dir)

	best_pval = sigma_results.groupby('random_seed').pval.max()
	best_pval = best_pval.to_frame()
	best_pval.reset_index(inplace=True, drop=False)
	best_pval.rename(columns={'pval': 'best_pval'}, inplace=True)

	smallest_mmd = sigma_results.groupby('random_seed').mmd.min()
	smallest_mmd = smallest_mmd.to_frame()
	smallest_mmd.reset_index(inplace=True, drop=False)
	smallest_mmd.rename(columns={'mmd': 'smallest_mmd'}, inplace=True)

	sigma_results = sigma_results.merge(best_pval, on ='random_seed')
	sigma_results = sigma_results.merge(smallest_mmd, on ='random_seed')

	filtered_results = all_results.merge(sigma_results, on=['random_seed', 'sigma', 'alpha'])

	filtered_results = filtered_results[
		(((filtered_results.pval >= pval) &  (filtered_results.best_pval >= pval)) | \
		((filtered_results.best_pval < pval) &  (filtered_results.mmd == filtered_results.smallest_mmd)))
		]

	best_pval_by_seed = filtered_results[['random_seed', 'pval']].copy()
	best_pval_by_seed = best_pval_by_seed.groupby('random_seed').pval.min()

	filtered_results.drop(['pval', 'best_pval'], inplace=True, axis=1)
	filtered_results.reset_index(drop=True, inplace=True)

	unique_filtered_results = filtered_results[['random_seed', 'sigma', 'alpha']].copy()
	unique_filtered_results.drop_duplicates(inplace=True)

	return get_optimal_model_classic(None, filtered_results, hparams)

def get_optimal_model_classic(results_file, filtered_results, hparams):
	if ((results_file is None) and (filtered_results is None)):
		raise ValueError("Need either filtered results or location of full results")
	elif results_file is None:
		all_results = filtered_results.copy()
	else: 
		all_results = pd.read_csv(results_file)

	columns_to_keep = hparams + ['random_seed', 'validation_pred_loss']
	best_loss = all_results[columns_to_keep]
	best_loss = best_loss.groupby('random_seed').validation_pred_loss.min()
	best_loss = best_loss.to_frame()


	best_loss.reset_index(drop=False, inplace=True)
	best_loss.rename(columns={'validation_pred_loss': 'min_validation_pred_loss'},
		inplace=True)
	all_results = all_results.merge(best_loss, on='random_seed')

	all_results = all_results[
		(all_results.validation_pred_loss == all_results.min_validation_pred_loss)
	]

	optimal_configs = all_results[['random_seed', 'hash']]

	# --- get the final results over all runs
	mean_results = all_results.mean(axis=0).to_frame()
	mean_results.rename(columns={0: 'mean'}, inplace=True)
	std_results = all_results.std(axis=0).to_frame()
	std_results.rename(columns={0: 'std'}, inplace=True)
	final_results = mean_results.merge(
		std_results, left_index=True, right_index=True
	)

	final_results = final_results.transpose()
	final_results_clean = reshape_results(final_results)

	return final_results_clean, optimal_configs


if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--results_file', '-results_file',
		help=("Pointer to the CSV file that has the results from"
		"cross validation"),
		type=str)

	parser.add_argument('--all_model_dir', '-all_model_dir',
		help=("comma separated list of all the directories "
			" that have the saved models and the config files"),
		type=str)

	parser.add_argument('--hparams', '-hparams',
		help=("comma separated list of the column names "
			"of the hyperparameter columsn in the CSV file"),
		type=str)

	parser.add_argument('--weighted_xv', '-weighted_xv',
		default='None',
		choices=['None', 'weighted_bal'],
		help=("Should we weight the cross validation metrics?"
			"This will follow the model hyperparameters if"
			"None is specified (i.e., if model is weigted, it"
			"will do weighted xv. If weighted_bal is"
			"specified, it will use the weights specified in"
			"the paper."),
		type=str)

	parser.add_argument('--dataset', '-dataset',
		default='waterbirds',
		choices=['waterbirds', 'chexpert'],
		help=("Which dataset?"),
		type=str)

	parser.add_argument('--data_dir', '-data_dir',
		help=("Directory that has all the experiment data"),
		type=str)

	parser.add_argument('--pval', '-pval',
		default=0.05,
		help=("the p-value above which we reject the null "
			"hypothesis that MMD = 0. Lower values prioritize"
			" robustness (i.e., lower MMD) over accuracy"),
		type=float)

	parser.add_argument('--kfolds', '-kfolds',
		default=3,
		help=("number of subgroups to divide the validation"
		" set into to estimated the variance of the MMD"),
		type=int)

	args = vars(parser.parse_args())
	get_optimal_model_two_step(**args)