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

"""Main file for running the chexpert experiment."""

from absl import app
from absl import flags

from chexpert import data_builder
from shared import train

FLAGS = flags.FLAGS
flags.DEFINE_string('skew_train', 'False', 'train on skewed data?')
flags.DEFINE_float('p_tr', .7, 'proportion of data used for training.')
flags.DEFINE_float('p_val', .25, 'proportion of the training data used for validation.')
flags.DEFINE_integer('pixel', 512, 'number of pixels in the image (i.e., res).')
flags.DEFINE_integer('Kfolds', 0, 'number of folds (i.e., batches) in validation set.'
	'If 0, it will be determined by batch_size')
flags.DEFINE_string('exp_dir', '/data/ddmg/slabs/dummy/',
										'Directory to save trained model in.')
flags.DEFINE_string('architecture', 'pretrained_densenet',
										'Architecture to use for training.')
flags.DEFINE_integer('batch_size', 16, 'batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('training_steps', 0, 'number of estimator training steps.'
										' If non-zero over rides the automatic value'
										' determined by num_epochs and batch_size')
flags.DEFINE_float('alpha', 1.0, 'Value for the cross prediction penalty')
flags.DEFINE_float('sigma', 1.0, 'Value for the MMD kernel bandwidth.')
flags.DEFINE_string('weighted_mmd', 'False',
											'use weighting when computing the mmd?.')
flags.DEFINE_string('balanced_weights', 'True',
											'balance weights? Set = True to get the weights used in the paper.')
flags.DEFINE_string('two_way_mmd', 'False',
											'two way mmd?.')
flags.DEFINE_float('dropout_rate', 0.0, 'Value for drop out rate')
flags.DEFINE_float('l2_penalty', 0.0,
									'L2 regularization penalty for final layer')
flags.DEFINE_integer('embedding_dim', 1000,
										'Dimension for the final embedding.')
flags.DEFINE_string('random_augmentation', 'False',
		'Augment data at training time using random transformations.')
flags.DEFINE_integer('random_seed', 0, 'random seed for tensorflow estimator')
flags.DEFINE_string('minimize_logits', 'False',
		'compute mmd wrt to logits if true and embedding if false.')
flags.DEFINE_string('gpuid', '0', 'Gpu id to run the model on.')
flags.DEFINE_string('debugger', 'False', 'debugger mode')

def main(argv):
	del argv

	def dataset_builder():
		return data_builder.build_input_fns(
			skew_train=FLAGS.skew_train,
			p_tr=FLAGS.p_tr,
			p_val=FLAGS.p_val,
			Kfolds=FLAGS.Kfolds,
			random_seed=FLAGS.random_seed)

	py1_y0_shift_list = [0.5, 0.9]

	train.train(
		exp_dir=FLAGS.exp_dir,
		dataset_builder=dataset_builder,
		architecture=FLAGS.architecture,
		training_steps=FLAGS.training_steps,
		pixel=FLAGS.pixel,
		num_epochs=FLAGS.num_epochs,
		batch_size=FLAGS.batch_size,
		Kfolds=FLAGS.Kfolds,
		alpha=FLAGS.alpha,
		sigma=FLAGS.sigma,
		balanced_weights=FLAGS.balanced_weights,
		weighted_mmd=FLAGS.weighted_mmd,
		two_way_mmd=FLAGS.two_way_mmd,
		dropout_rate=FLAGS.dropout_rate,
		l2_penalty=FLAGS.l2_penalty,
		embedding_dim=FLAGS.embedding_dim,
		random_augmentation=FLAGS.random_augmentation,
		random_seed=FLAGS.random_seed,
		minimize_logits=FLAGS.minimize_logits,
		py1_y0_shift_list=py1_y0_shift_list,
		debugger=FLAGS.debugger)

if __name__ == '__main__':
	app.run(main)
