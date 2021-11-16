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

"""Commonly used neural network architectures."""

# NOTE:see batch norm issues here https://github.com/keras-team/keras/pull/9965
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121

def create_architecture(params):
	# architectures without random augmentation
	if params['random_augmentation'] == 'False':
		if params['architecture'] == 'simple':
			net = SimpleConvolutionNet(
				dropout_rate=params["dropout_rate"],
				l2_penalty=params["l2_penalty"],
				embedding_dim=params["embedding_dim"])
		if params['architecture'] == 'pretrained_resnet':
			net = PretrainedResNet50(
				embedding_dim=params["embedding_dim"],
				l2_penalty=params["l2_penalty"])
		if params['architecture'] == 'pretrained_resnet_random':
			net = RandomResNet50(
				embedding_dim=params["embedding_dim"],
				l2_penalty=params["l2_penalty"])
		if params['architecture'] == 'pretrained_resnet101':
			net = PretrainedResNet101(
				embedding_dim=params["embedding_dim"],
				l2_penalty=params["l2_penalty"])
		if params['architecture'] == 'pretrained_densenet':
				net = PretrainedDenseNet121(
					embedding_dim=params["embedding_dim"],
					l2_penalty=params["l2_penalty"])

	# architectures with random augmentataion
	if params['random_augmentation'] == 'True':
		if (params['architecture'] == 'pretrained_resnet'):
			net = PretrainedResNet50_RandomAugmentation(
				embedding_dim=params["embedding_dim"],
				l2_penalty=params["l2_penalty"])
		if (params['architecture'] == 'pretrained_densenet'):
			net = PretrainedDenseNet121_RandomAugmentation(
				embedding_dim=params["embedding_dim"],
				l2_penalty=params["l2_penalty"])

	return net

class PretrainedResNet50(tf.keras.Model):
	"""Resnet 50 pretrained on imagenet."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False):
		# Note: if embedding_dim = -1, the embedding dimension is decided
		# based on the architecture (2048 for resnet)
		super(PretrainedResNet50, self).__init__()
		self.embedding_dim = embedding_dim

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x


class PretrainedResNet50_RandomAugmentation(tf.keras.Model):
	"""Resnet 50 pretrained on imagenet with random augmentation"""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedResNet50_RandomAugmentation, self).__init__()
		# Note: if embedding_dim = -1, the embedding dimension is decided
		# based on the architecture (2048 for resnet)
		self.embedding_dim = embedding_dim

		self.data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
		])

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		if training:
			x = self.data_augmentation(inputs, training=training)
		else:
			x = inputs
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x

class RandomResNet50(tf.keras.Model):
	"""Randomly initialized resnet 50."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0):
		super(RandomResNet50, self).__init__()
		self.embedding_dim = embedding_dim
		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights=None)
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		regularizer = tf.keras.regularizers.l2(l2_penalty)
		for layer in self.resenet.layers:
			if hasattr(layer, 'kernel'):
				self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x


class PretrainedDenseNet121(tf.keras.Model):
	"""Densenet121 pretrained on imagenet."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedDenseNet121, self).__init__()
		self.embedding_dim = embedding_dim

		self.densenet = DenseNet121(include_top=False, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.densenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.densenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x



class PretrainedDenseNet121_RandomAugmentation(tf.keras.Model):
	"""Densenet121 pretrained on imagenet with random augmentataions"""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedDenseNet121_RandomAugmentation, self).__init__()
		self.embedding_dim = embedding_dim

		self.data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
		])

		self.densenet = DenseNet121(include_top=False, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.densenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		# TODO fix this
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		if training:
			x = self.data_augmentation(inputs, training=training)
		else:
			x = inputs
		x = self.densenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x



class PretrainedResNet101(tf.keras.Model):
	"""resent 101 pretrained on imagenet"""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0):
		super(PretrainedResNet101, self).__init__()
		self.embedding_dim = embedding_dim
		self.resenet = tf.keras.applications.ResNet101(include_top=False,
			layers=tf.keras.layers, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		regularizer = tf.keras.regularizers.l2(l2_penalty)
		for layer in self.resenet.layers:
			if hasattr(layer, 'kernel'):
				self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x


class SimpleConvolutionNet(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, dropout_rate=0.0, l2_penalty=0.0, embedding_dim=1000):
		super(SimpleConvolutionNet, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
		self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu")
		self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
		self.dropout = tf.keras.layers.Dropout(dropout_rate)

		self.flatten1 = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			embedding_dim,
			activation="relu",
			kernel_regularizer=tf.keras.regularizers.L2(l2=l2_penalty),
			name="Z")
		self.dense2 = tf.keras.layers.Dense(1)

	def call(self, inputs, training=False):
		z = self.conv1(inputs)
		z = self.conv2(z)
		z = self.maxpool1(z)
		if training:
			z = self.dropout(z, training=training)
		z = self.flatten1(z)
		z = self.dense1(z)
		return self.dense2(z), z
