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

"""Evaluation metrics for the main method."""
import copy

from shared import losses

import tensorflow as tf

def compute_pred_loss(labels, logits, sample_weights, params):

	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)
	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	if params['weighted_mmd'] == 'False':
		return tf.reduce_mean(individual_losses)

	weighted_loss = sample_weights * individual_losses
	weighted_loss = tf.math.divide_no_nan(
		tf.reduce_sum(weighted_loss),
		tf.reduce_sum(sample_weights)
	)
	return weighted_loss


def compute_loss(labels, logits, z_pred, sample_weights,
	sample_weights_pos, sample_weights_neg, params):
	if params['weighted_mmd'] == 'False':
		return compute_loss_unweighted(labels, logits, z_pred, params)
	return compute_loss_weighted(labels, logits, z_pred,
		sample_weights, sample_weights_pos, sample_weights_neg,  params)

def compute_loss_weighted(labels, logits, z_pred, sample_weights,
	sample_weights_pos, sample_weights_neg, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	# --- Prediction loss
	weighted_loss = sample_weights * individual_losses
	weighted_loss = tf.math.divide_no_nan(
		tf.reduce_sum(weighted_loss),
		tf.reduce_sum(sample_weights)
	)

	# --- MMD loss
	if params['minimize_logits'] == 'True':
		embedding_features = logits
	else:
		embedding_features = z_pred

	other_label_inds = [
		lab_ind for lab_ind in range(labels.shape[1])
		if lab_ind != params["label_ind"]
	]

	weighted_mmd_vals = []
	for lab_ind in other_label_inds:
		mmd_val = losses.mmd_loss(
			embedding=embedding_features,
			auxiliary_labels=labels[:, lab_ind],
			weights_pos=sample_weights_pos,
			weights_neg=sample_weights_neg,
			params=params)
		weighted_mmd_vals.append(mmd_val[0])

	weighted_mmd = tf.concat(weighted_mmd_vals, axis=0)

	return weighted_loss, weighted_mmd


def compute_loss_unweighted(labels, logits, z_pred, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	individual_losses = tf.keras.losses.binary_crossentropy(
		y_main, logits, from_logits=True)

	# --- Prediction loss
	unweighted_loss = tf.reduce_mean(individual_losses)

	# --- MMD loss
	if params['minimize_logits'] == 'True':
		embedding_features = logits
	else:
		embedding_features = z_pred

	other_label_inds = [
		lab_ind for lab_ind in range(labels.shape[1])
		if lab_ind != params["label_ind"]
	]

	unweighted_mmd_vals = []
	for lab_ind in other_label_inds:
		mmd_val = losses.mmd_loss(
			embedding=embedding_features,
			auxiliary_labels=labels[:, lab_ind],
			weights_pos=None,
			weights_neg=None,
			params=params)
		unweighted_mmd_vals.append(mmd_val[0])

	unweighted_mmd = tf.concat(unweighted_mmd_vals, axis=0)

	return unweighted_loss, unweighted_mmd

def get_prediction_by_group(labels, predictions):

	mean_prediction_dict = {}

	labels11_mask = tf.where(labels[:, 0] * labels[:, 1])
	mean_prediction_dict['mean_pred_11'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels11_mask)
	)

	labels10_mask = tf.where(labels[:, 0] * (1.0 - labels[:, 1]))
	mean_prediction_dict['mean_pred_10'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels10_mask)
	)

	labels01_mask = tf.where((1.0 - labels[:, 0]) * labels[:, 1])
	mean_prediction_dict['mean_pred_01'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels01_mask)
	)

	labels00_mask = tf.where((1.0 - labels[:, 0]) * (1.0 - labels[:, 1]))
	mean_prediction_dict['mean_pred_00'] = tf.compat.v1.metrics.mean(
		tf.gather(predictions, labels00_mask)
	)

	return mean_prediction_dict


def auroc(labels, predictions):
	""" Computes AUROC """
	auc_metric = tf.keras.metrics.AUC(name="auroc")
	auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric

def get_eval_metrics_dict(labels, predictions, sample_weights,
	sample_weights_pos, sample_weights_neg, sigma_list, params):
	y_main = tf.expand_dims(labels[:, params["label_ind"]], axis=-1)

	eval_metrics_dict = {}

	# -- the "usual" evaluation metrics
	eval_metrics_dict['accuracy'] = tf.compat.v1.metrics.accuracy(
		labels=y_main, predictions=predictions["classes"])

	eval_metrics_dict["auc"] = auroc(
		labels=y_main, predictions=predictions["probabilities"])

	# -- Mean predictions for each group
	mean_prediction_by_group = get_prediction_by_group(labels,
		predictions["probabilities"])

	return {**eval_metrics_dict, **mean_prediction_by_group}


