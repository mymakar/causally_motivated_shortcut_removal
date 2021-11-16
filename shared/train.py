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

"""Main training protocol used for structured label prediction models."""
import os
import pickle
import copy

from shared import architectures
from shared import train_utils
from shared import evaluation_metrics


import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import array_ops, variable_scope
from tensorflow.python.framework import ops, dtypes

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def serving_input_fn():
	"""Serving function to facilitate model saving."""
	feat = array_ops.placeholder(dtype=dtypes.float32)
	return tf.estimator.export.TensorServingInputReceiver(features=feat,
		receiver_tensors=feat)

def model_fn(features, labels, mode, params):
	""" Main training function ."""

	net = architectures.create_architecture(params)

	training_state = mode == tf.estimator.ModeKeys.TRAIN
	logits, zpred = net(features, training=training_state)
	ypred = tf.nn.sigmoid(logits)

	predictions = {
		"classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
		"logits": logits,
		"probabilities": ypred,
		"embedding": zpred
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				"classify": tf.estimator.export.PredictOutput(predictions)
			})

	if labels['labels'].shape[1] != 2:
		raise NotImplementedError('Only 2 labels supported for now')

	sample_weights, sample_weights_pos, sample_weights_neg = train_utils.extract_weights(labels, params)
	labels = tf.identity(labels['labels'])

	if mode == tf.estimator.ModeKeys.EVAL:
		main_eval_metrics = {}

		# -- main loss components
		eval_pred_loss, eval_mmd_loss = evaluation_metrics.compute_loss(labels, logits, zpred,
			sample_weights, sample_weights_pos, sample_weights_neg, params)

		main_eval_metrics['pred_loss'] = tf.compat.v1.metrics.mean(eval_pred_loss)
		main_eval_metrics['mmd'] = tf.compat.v1.metrics.mean(eval_mmd_loss)

		loss = eval_pred_loss + params["alpha"] * eval_mmd_loss

		# -- additional eval metrics
		additional_eval_metrics = evaluation_metrics.get_eval_metrics_dict(
			labels, predictions, sample_weights,
			sample_weights_pos, sample_weights_neg, params)

		eval_metrics = {**main_eval_metrics, **additional_eval_metrics}

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=None, eval_metric_ops=eval_metrics)

	if mode == tf.estimator.ModeKeys.TRAIN:
		opt = tf.keras.optimizers.Adam()
		global_step = tf.compat.v1.train.get_global_step()

		ckpt = tf.train.Checkpoint(
			step=global_step, optimizer=opt, net=net)

		with tf.GradientTape() as tape:
			logits, zpred = net(features, training=training_state)
			ypred = tf.nn.sigmoid(logits)

			prediction_loss, mmd_loss = evaluation_metrics.compute_loss(labels, logits, zpred,
				sample_weights, sample_weights_pos, sample_weights_neg, params)

			regularization_loss = tf.reduce_sum(net.losses)
			loss = regularization_loss + prediction_loss + params["alpha"] * mmd_loss


		variables = net.trainable_variables
		gradients = tape.gradient(loss, variables)

		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=tf.group(
				opt.apply_gradients(zip(gradients, variables)),
				ckpt.step.assign_add(1)))

def train(exp_dir,
					dataset_builder,
					architecture,
					training_steps,
					pixel,
					num_epochs,
					batch_size,
					Kfolds,
					alpha,
					sigma,
					weighted_mmd,
					balanced_weights,
					dropout_rate,
					l2_penalty,
					embedding_dim,
					random_augmentation,
					random_seed,
					minimize_logits,
					py1_y0_shift_list=None):
	"""Trains the estimator."""

	input_fns = dataset_builder()
	training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater = input_fns
	steps_per_epoch = int(training_data_size / batch_size)

	params = {
		"pixel": pixel,
		"architecture": architecture,
		"num_epochs": num_epochs,
		"batch_size": batch_size,
		"steps_per_epoch": steps_per_epoch,
		"Kfolds": Kfolds,
		"alpha": alpha,
		"sigma": sigma,
		"weighted_mmd": weighted_mmd,
		"balanced_weights": balanced_weights,
		"dropout_rate": dropout_rate,
		"l2_penalty": l2_penalty,
		"embedding_dim": embedding_dim,
		"random_augmentation": random_augmentation,
		"minimize_logits": minimize_logits,
		"label_ind": 0
	}

	run_config = tf.estimator.RunConfig(
		tf_random_seed=random_seed,
		save_checkpoints_steps=1000,
		keep_checkpoint_max=2)

	est = tf.estimator.Estimator(
		model_fn, model_dir=exp_dir, params=params, config=run_config)

	if training_steps == 0:
		training_steps = int(params['num_epochs'] * steps_per_epoch)

	est.train(train_input_fn, steps=training_steps)

	validation_results = est.evaluate(valid_input_fn)
	sym_results = {"validation": validation_results}

	# ---- Get test results
	if py1_y0_shift_list is not None:
		# -- during testing, we dont have access to labels/weights
		test_params = copy.deepcopy(params)
		test_params['weighted_mmd'] = 'False'
		test_params['balanced_weights'] = 'False'
		for py in py1_y0_shift_list:
			eval_input_fn = eval_input_fn_creater(py, test_params, asym=False)
			distribution_results = est.evaluate(eval_input_fn, steps=1e5)
			sym_results[f'shift_{py}'] = distribution_results

	# save results
	savefile = f"{exp_dir}/performance.pkl"
	sym_results = train_utils.flatten_dict(sym_results)
	pickle.dump(sym_results, open(savefile, "wb"))


	# save model
	est.export_saved_model(f'{exp_dir}/saved_model', serving_input_fn)

