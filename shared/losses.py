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

"""Library of non-typical losses."""

import tensorflow as tf
import tensorflow_probability as tfp


def mmd_loss(embedding, auxiliary_labels, weights_pos, weights_neg, params):
	""" Computes mmd loss, weighted or unweighted """
	if weights_pos is None:
		return mmd_loss_unweighted(embedding, auxiliary_labels, params)
	return mmd_loss_weighted(embedding, auxiliary_labels,
		weights_pos, weights_neg, params)

def mmd_loss_unweighted(embedding, auxiliary_labels, params):
	r"""Computes unweighted MMD loss between embeddings of groups defined by label.

	Maximum Mean Discrepancy (MMD) is an integrated probability metric.
	It measures the distance between two probability distributions. In this
	setting, we measure (and penalize) the distance between the probability
	distributions of the embeddings of group 0 (where auxiliary_labels ==0), and
	the emeddings of group 1 (where auxiliary_labels ==1). The specific equation
	for computing the MMD is:

	MMD^2(P, Q)= || \E{\phi_sigma(x)} - \E{\phi_sigma(y)} ||^2
						= \E{ K_sigma(x, x) } + \E{ K_sigma(y, y) } - 2 \E{ K_sigma(x, y)},

	where K_sigma = <\phi_sigma(x), \phi_sigma(y)>,is a kernel function,
	in this case a radial basis kernel, with bandwidth sigma.

	For our main approach, we penalize the mmd_loss (encourage the distance to be
	small i.e., encourage the two distributions to be close, which is roughly
	equivalent to an adversarial setting: by forcing MMD to be small, an
	adversary cannot distinguish between the two groups solely based on the
	embedding. This also implies that cross-prediction (predicting
	auxiliary_labels using embedding) is penalized.

	Args:
		embedding: tensor with learned embedding
		auxiliary_labels: tensor with label defining 2 groups
		params: dictionary of parameters, must include sigma, the
			kernel bandwidth
	Returns:
		mmd_val: MMD between embeddings of the two groups defined by label
		pos_kernel_mean: not used, just for debugging. kernel mean of the
			group with aux label = 1
		neg_kernel_mean: not used, just for debugging. kernel mean of the
			group with aux label = 0
		pos_neg_kernel_mean: not used, just for debugging.
			mean of the kernel between pos & neg examples
		neg_pos_kernel_mean: not used, just for debugging.
			mean of the kernel between neg & pos examples
	"""
	sigma = params['sigma']
	del params

	kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
		amplitude=1.0, length_scale=sigma)

	kernel_mat = kernel.matrix(embedding, embedding)

	if len(auxiliary_labels.shape) == 1:
		auxiliary_labels = tf.expand_dims(auxiliary_labels, axis=-1)

	pos_mask = tf.matmul(auxiliary_labels, tf.transpose(auxiliary_labels))
	neg_mask = tf.matmul(1.0 - auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	pos_neg_mask = tf.matmul(auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))

	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_mask * kernel_mat), tf.reduce_sum(pos_mask))
	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_mask * kernel_mat), tf.reduce_sum(neg_mask))
	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_mask * kernel_mat), tf.reduce_sum(pos_neg_mask))

	mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
	mmd_val = tf.maximum(0.0, mmd_val)

	return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean


def mmd_loss_weighted(embedding, auxiliary_labels, weights_pos,
	weights_neg, params):
	r""" NOTE: this function is not used, alternative way to compute the weighted
	mmd. weights the examples both ways (rows and columns).

	Computes MMD loss between embeddings of groups defined by label.

	Maximum Mean Discrepancy (MMD) is an integrated probability metric.
	It measures the distance between two probability distributions. In this
	setting, we measure (and penalize) the distance between the probability
	distributions of the embeddings of group 0 (where auxiliary_labels ==0), and
	the emeddings of group 1 (where auxiliary_labels ==1). The specific equation
	for computing the MMD is:

	MMD^2(P, Q)= || \E{\phi_sigma(x)} - \E{\phi_sigma(y)} ||^2
						= \E{ K_sigma(x, x) } + \E{ K_sigma(y, y) } - 2 \E{ K_sigma(x, y)},

	where K_sigma = <\phi_sigma(x), \phi_sigma(y)>,is a kernel function,
	in this case a radial basis kernel, with bandwidth sigma.

	For our main approach, we penalize the mmd_loss (encourage the distance to be
	small i.e., encourage the two distributions to be close, which is roughly
	equivalent to an adversarial setting: by forcing MMD to be small, an
	adversary cannot distinguish between the two groups solely based on the
	embedding. This also implies that cross-prediction (predicting
	auxiliary_labels using embedding) is penalized

	Args:
		embedding: tensor with learned embedding
		auxiliary_labels: tensor with label defining 2 groups
		weights_pos: tensor with weights for the positive group
		weights_neg: tensor with weights for the negative group
		params: dictionary of parameters, must include sigma, the
			kernel bandwidth
	Returns:
		mmd_val: MMD between embeddings of the two groups defined by label
		pos_kernel_mean: not used, just for debugging. kernel mean of the
			group with aux label = 1
		neg_kernel_mean: not used, just for debugging. kernel mean of the
			group with aux label = 0
		pos_neg_kernel_mean: not used, just for debugging.
			mean of the kernel between pos & neg examples
		neg_pos_kernel_mean: not used, just for debugging.
			mean of the kernel between neg & pos examples
	"""
	sigma = params['sigma']
	kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
		amplitude=1.0, length_scale=sigma)

	kernel_mat = kernel.matrix(embedding, embedding)

	if len(auxiliary_labels.shape) == 1:
		auxiliary_labels = tf.expand_dims(auxiliary_labels, axis=-1)

	pos_mask = tf.matmul(auxiliary_labels, tf.transpose(auxiliary_labels))
	neg_mask = tf.matmul(1.0 - auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	pos_neg_mask = tf.matmul(auxiliary_labels,
		tf.transpose(1.0 - auxiliary_labels))
	neg_pos_mask = tf.matmul((1.0 - auxiliary_labels),
		tf.transpose(auxiliary_labels))

	pos_kernel_mean = kernel_mat * pos_mask
	pos_kernel_mean = pos_kernel_mean * tf.transpose(weights_pos)

	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_kernel_mean, axis=1),
		tf.reduce_sum(weights_pos))

	pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_kernel_mean * tf.squeeze(weights_pos)),
		tf.reduce_sum(weights_pos))

	neg_kernel_mean = kernel_mat * neg_mask
	neg_kernel_mean = neg_kernel_mean * tf.transpose(weights_neg)

	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_kernel_mean, axis=1),
		tf.reduce_sum(weights_neg))

	neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_kernel_mean * tf.squeeze(weights_neg)),
		tf.reduce_sum(weights_neg))

	neg_pos_kernel_mean = kernel_mat * neg_pos_mask
	neg_pos_kernel_mean = neg_pos_kernel_mean * tf.transpose(weights_pos)

	neg_pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_pos_kernel_mean, axis=1),
		tf.reduce_sum(weights_pos))

	neg_pos_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(neg_pos_kernel_mean * tf.squeeze(weights_neg)),
		tf.reduce_sum(weights_neg))

	pos_neg_kernel_mean = kernel_mat * pos_neg_mask
	pos_neg_kernel_mean = pos_neg_kernel_mean * tf.transpose(weights_neg)

	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_kernel_mean, axis=1),
		tf.reduce_sum(weights_neg))

	pos_neg_kernel_mean = tf.math.divide_no_nan(
		tf.reduce_sum(pos_neg_kernel_mean * tf.squeeze(weights_pos)),
		tf.reduce_sum(weights_pos))

	mmd_val = pos_kernel_mean + neg_kernel_mean - (
		pos_neg_kernel_mean + neg_pos_kernel_mean)
	mmd_val = tf.maximum(0.0, mmd_val)

	return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean




