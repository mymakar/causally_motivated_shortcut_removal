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

"""Utility functions to support the main training algorithm"""
import glob
import os
import tensorflow as tf

def extract_weights(labels, params):
	""" Extracts the weights from the labels dictionary. """

	if (params['weighted_mmd'] == 'True') and (params['balanced_weights'] == 'True'):
		sample_weights_pos = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 0]), axis=-1)
		sample_weights_neg = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 1]), axis=-1)
		sample_weights = tf.expand_dims(tf.identity(labels['balanced_weights'][:, 2]), axis=-1)
	else:
		sample_weights_pos = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 0]), axis=-1)
		sample_weights_neg = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 1]), axis=-1)
		sample_weights = tf.expand_dims(tf.identity(labels['unbalanced_weights'][:, 2]), axis=-1)


	return sample_weights, sample_weights_pos, sample_weights_neg

def flatten_dict(dd, separator='_', prefix=''):
	""" Flattens the dictionary with eval metrics """
	return {
		prefix + separator + k if prefix else k: v
		for kk, vv in dd.items()
		for k, v in flatten_dict(vv, separator, kk).items()
	} if isinstance(dd,
		dict) else {prefix: dd}


def cleanup_directory(directory):
	"""Deletes all files within directory and its subdirectories.

	Args:
		directory: string, the directory to clean up
	Returns:
		None
	"""
	if os.path.exists(directory):
		files = glob.glob(f"{directory}/*", recursive=True)
		for f in files:
			if os.path.isdir(f):
				os.system(f"rm {f}/* ")
			else:
				os.system(f"rm {f}")


