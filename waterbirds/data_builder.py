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

"""Functions to create the waterbirds datasets.

Code based on https://github.com/kohpangwei/group_DRO/blob/master/
	dataset_scripts/generate_waterbirds.py
"""
import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = '/path/to/processed_places_data/'
IMAGE_DIR = '/path/to/CUB_200_2011/'
SEGMENTATION_DIR = '/path/to/segmentations/'


NUM_PLACE_IMAGES_CLEAN = 8000
WATER_IMG_DIR_CLEAN = 'water_easy'
LAND_IMG_DIR_CLEAN = 'land_easy'

NUM_PLACE_IMAGES = 10000
WATER_IMG_DIR = 'water'
LAND_IMG_DIR = 'land'

WATERBIRD_LIST = [
	'Albatross', 'Auklet', 'Cormorant', 'Frigatebird', 'Fulmar', 'Gull', 'Jaeger',
	'Kittiwake', 'Pelican', 'Puffin', 'Tern', 'Gadwall', 'Grebe', 'Mallard',
	'Merganser', 'Guillemot', 'Pacific_Loon'
]

def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img


def read_decode_png(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=1)
	return img


def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label


def map_to_image_label(x, pixel):

	weights_included = len(x) > 5

	bird_image = x[0]
	bird_segmentation = x[1]
	background_image = x[2]
	y0 = x[3]
	y1 = x[4]

	if weights_included:
		unbalanced_weights_pos = x[5]
		unbalanced_weights_neg = x[6]
		unbalanced_weights_both = x[7]

		balanced_weights_pos = x[8]
		balanced_weights_neg = x[9]
		balanced_weights_both = x[10]

	# decode images
	bird_image = read_decode_jpg(bird_image)
	bird_segmentation = read_decode_png(bird_segmentation)
	background_image = read_decode_jpg(background_image)

	# get binary segmentation
	bird_segmentation = tf.math.round(bird_segmentation / 255)
	bird_segmentation = tf.cast(bird_segmentation, tf.uint8)

	# resize the background image
	bkgrd_resized = tf.image.resize(background_image,
		(tf.shape(bird_image)[0], tf.shape(bird_image)[1]))
	bkgrd_resized = tf.cast(bkgrd_resized, tf.uint8)

	# get the masked image
	img = bird_image * bird_segmentation + bkgrd_resized * (1 - bird_segmentation)

	# resize, rescale  image
	img = tf.image.resize(img, (pixel, pixel))
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	labels = tf.concat([y0, y1], axis=0)

	# get the weights
	if weights_included:
		unbalanced_weights_pos = decode_number(unbalanced_weights_pos)
		unbalanced_weights_neg = decode_number(unbalanced_weights_neg)
		unbalanced_weights_both = decode_number(unbalanced_weights_both)

		unbalanced_weights = tf.concat([unbalanced_weights_pos,
			unbalanced_weights_neg, unbalanced_weights_both], axis=0)

		balanced_weights_pos = decode_number(balanced_weights_pos)
		balanced_weights_neg = decode_number(balanced_weights_neg)
		balanced_weights_both = decode_number(balanced_weights_both)

		balanced_weights = tf.concat([balanced_weights_pos,
			balanced_weights_neg, balanced_weights_both], axis=0)

	else:
		unbalanced_weights = None
		balanced_weights = None

	labels_and_weights = {
		'labels': labels,
		'unbalanced_weights': unbalanced_weights,
		'balanced_weights': balanced_weights,
	}

	return img, labels_and_weights


def get_bird_type(x):
	bird_type = [
		water_bird_name in x['img_filename'] for water_bird_name in WATERBIRD_LIST
	]
	bird_type = max(bird_type) * 1
	return bird_type


def get_weights(data_frame):

	water_back_water_bird_weight = np.sum(
		data_frame.y0 * data_frame.y1) / np.sum(data_frame.y1)
	water_back_land_bird_weight = np.sum(
		(1.0 - data_frame.y0) * data_frame.y1) / np.sum(data_frame.y1)

	land_back_water_bird_weight = np.sum(
		data_frame.y0 * (1.0 - data_frame.y1)) / np.sum((1.0 - data_frame.y1))
	land_back_land_bird_weight = np.sum(
		(1.0 - data_frame.y0) * (1.0 - data_frame.y1)) / np.sum(
		(1.0 - data_frame.y1))

	# -- positive weights
	data_frame['weights_pos'] = data_frame.y0 * water_back_water_bird_weight + \
		(1.0 - data_frame.y0) * water_back_land_bird_weight
	data_frame['weights_pos'] = 1.0 / data_frame['weights_pos']
	data_frame['weights_pos'] = data_frame.y1 * data_frame['weights_pos']

	assert data_frame.weights_pos.isin([np.nan, np.inf, -np.inf]).sum() == 0

	data_frame['balanced_weights_pos'] = np.mean(data_frame.y0) * \
		data_frame.y0 * data_frame.weights_pos + \
		np.mean(1.0 - data_frame.y0) * (1.0 - data_frame.y0) * data_frame.weights_pos

	# -- negative weights
	data_frame['weights_neg'] = data_frame.y0 * land_back_water_bird_weight + \
		(1.0 - data_frame.y0) * land_back_land_bird_weight
	data_frame['weights_neg'] = 1.0 / data_frame['weights_neg']
	data_frame['weights_neg'] = (1.0 - data_frame.y1) * data_frame['weights_neg']

	assert data_frame.weights_neg.isin([np.nan, np.inf, -np.inf]).sum() == 0

	data_frame['balanced_weights_neg'] = np.mean(data_frame.y0) * \
		data_frame.y0 * data_frame.weights_neg + \
		np.mean(1.0 - data_frame.y0) * (1.0 - data_frame.y0) * data_frame.weights_neg

	# aggregate weights
	data_frame['weights'] = data_frame['weights_pos'] + data_frame['weights_neg']
	data_frame['balanced_weights'] = data_frame['balanced_weights_pos'] + \
		data_frame['balanced_weights_neg']

	return data_frame


def create_images_labels(bird_data_frame, water_images, land_images, py1_y0=1,
	pflip0=.1, pflip1=.1, clean_back='False', rng=None):
	if rng is None:
		rng = np.random.RandomState(0)

	if clean_back == 'True':
		water_img_dir = WATER_IMG_DIR_CLEAN
		land_img_dir = LAND_IMG_DIR_CLEAN
		num_place_images = NUM_PLACE_IMAGES_CLEAN
	else:
		water_img_dir = WATER_IMG_DIR
		land_img_dir = LAND_IMG_DIR
		num_place_images = NUM_PLACE_IMAGES

	# -- add noise to bird type
	flip0 = rng.choice(bird_data_frame.shape[0],
		size=int(pflip0 * bird_data_frame.shape[0]), replace=False).tolist()
	bird_data_frame.y0.loc[flip0] = 1 - bird_data_frame.y0.loc[flip0]

	# -- get background type
	bird_data_frame['y1'] = rng.binomial(1,
		bird_data_frame.y0 * py1_y0 + (1 - bird_data_frame.y0) * (1.0 - py1_y0))


	# -- add noise
	flip1 = rng.choice(bird_data_frame.shape[0],
		size=int(pflip1 * bird_data_frame.shape[0]), replace=False).tolist()
	bird_data_frame.y1.loc[flip1] = 1 - bird_data_frame.y1.loc[flip1]

	# -- randomly pick land and water images
	water_image_ids = rng.choice(water_images,
		size=int(bird_data_frame.y1.sum()), replace=False)
	water_backgrounds = [
		f'{water_img_dir}/image_{img_id}.jpg' for img_id in water_image_ids
	]

	land_image_ids = rng.choice(land_images,
		size=int((1 - bird_data_frame.y1).sum()), replace=False)
	land_backgrounds = [
		f'{land_img_dir}/image_{img_id}.jpg' for img_id in land_image_ids
	]

	bird_data_frame['background_filename'] = ''
	bird_data_frame.background_filename[(
		bird_data_frame.y1 == 1)] = water_backgrounds
	bird_data_frame.background_filename[(
		bird_data_frame.y1 == 0)] = land_backgrounds

	return bird_data_frame, water_image_ids, land_image_ids


def save_created_data(data_frame, experiment_directory, filename):
	data_frame['img_filename'] = data_frame['img_filename'].str[:-3]
	txt_df = f'{IMAGE_DIR}/images/' + data_frame.img_filename + 'jpg' + \
		',' + SEGMENTATION_DIR + data_frame.img_filename + 'png' + \
		',' + f'{DATA_DIR}/places_data/' + data_frame.background_filename + \
		',' + data_frame.y0.astype(str) + \
		',' + data_frame.y1.astype(str) + \
		',' + data_frame.weights_pos.astype(str) + \
		',' + data_frame.weights_neg.astype(str) + \
		',' + data_frame.weights.astype(str) + \
		',' + data_frame.balanced_weights_pos.astype(str) + \
		',' + data_frame.balanced_weights_neg.astype(str) + \
		',' + data_frame.balanced_weights.astype(str)

	txt_df.to_csv(f'{experiment_directory}/{filename}.txt',
		index=False)


def load_created_data(experiment_directory, py1_y0_s):

	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt').values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = pd.read_csv(
		f'{experiment_directory}/validation.txt').values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	if py1_y0_s is None:
		return train_data, validation_data, None

	test_data_dict = {}
	for py1_y0_s_val in py1_y0_s:
		test_data = pd.read_csv(
			f'{experiment_directory}/test_shift{py1_y0_s_val}.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		test_data_dict[py1_y0_s_val] = test_data

	return train_data, validation_data, test_data_dict

def create_save_waterbird_lists(experiment_directory, py0=0.8, p_tr=.7,
	p_val=p_val, py1_y0=1,py1_y0_s=.5, pflip0=.1, pflip1=.1,
	clean_back='False', random_seed=None):

	if (py0 != 0.8) and (py0 != 0.5):
		raise NotImplementedError("Only accepting values of 0.8 and 0.5 for now")

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	if clean_back == 'True':
		num_place_images = NUM_PLACE_IMAGES_CLEAN
	else:
		num_place_images = NUM_PLACE_IMAGES

	# --- read in all bird image filenames

	df = pd.read_csv(f'{IMAGE_DIR}/images.txt', sep=" ", header=None,
		names=['img_id', 'img_filename'], index_col='img_id')
	df = df.sample(frac=1, random_state=random_seed)
	df.reset_index(inplace=True, drop=True)

	df = df[((
		df.img_filename.str.contains('Gull')) | (
		df.img_filename.str.contains('Warbler')
	))]

	df.reset_index(inplace=True, drop=True)

	# -- get bird type
	df['y0'] = df.apply(get_bird_type, axis=1)

	if py0 == 0.5:
		land_birds_to_keep = rng.choice(df[(df.y0 == 0)].index,
			size=df.y0.sum(), replace=False).tolist()
		water_birds_to_keep = df[(df.y0 == 1)].index.tolist()
		birds_to_keep = land_birds_to_keep + water_birds_to_keep
		birds_to_keep = rng.choice(birds_to_keep,size=len(birds_to_keep),
			replace=False).tolist()
		df = df.iloc[birds_to_keep]
		df.reset_index(inplace=True, drop=True)

	train_val_ids = rng.choice(df.shape[0],
		size=int(p_tr * df.shape[0]), replace=False).tolist()
	df['train_valid_ids'] = 0
	df.train_valid_ids.loc[train_val_ids] = 1

	# --- get the train and validation data
	train_valid_df = df[(df.train_valid_ids == 1)].reset_index(drop=True)

	train_valid_df, used_water_img_ids, used_land_img_ids = create_images_labels(
		train_valid_df, num_place_images, num_place_images, py1_y0=py1_y0,
		pflip0=pflip0, pflip1=pflip1, clean_back=clean_back)


	# --- create train validation split
	train_ids = rng.choice(train_valid_df.shape[0],
		size=int((1-p_val) * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1

	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	train_df = get_weights(train_df)
	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	valid_df = get_weights(valid_df)

	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename='validation')

	# --- create + save test data
	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)

	available_water_ids = list(set(range(num_place_images)) - set(
		used_water_img_ids))
	available_land_ids = list(set(range(num_place_images)) - set(
		used_land_img_ids))

	for py1_y0_s_val in py1_y0_s:
		curr_test_df = test_df.copy()
		curr_test_df, _, _ = create_images_labels(
			curr_test_df, available_water_ids, available_land_ids, py1_y0=py1_y0_s_val,
			pflip0=pflip0, pflip1=pflip1, clean_back=clean_back)

		curr_test_df = get_weights(curr_test_df)
		save_created_data(curr_test_df, experiment_directory=experiment_directory,
			filename=f'test_shift{py1_y0_s_val}')


def build_input_fns(p_tr=.7, p_val=0.25, py0=0.8, py1_y0=1, py1_y0_s=.5, pflip0=.1,
	pflip1=.1, Kfolds=0, clean_back='False', random_seed=None):

	if clean_back == 'True':
		experiment_directory = (f'{DATA_DIR}/experiment_data/'
			f'cleanback_rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')
	else:
		experiment_directory = (f'{DATA_DIR}/experiment_data/'
			f'rs{random_seed}_py0{py0}_py1_y0{py1_y0}_pfilp{pflip0}')

	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/train.txt'):
		if not os.path.exists(experiment_directory):
			os.mkdir(experiment_directory)

		create_save_waterbird_lists(
			experiment_directory=experiment_directory,
			py0=py0,
			p_tr=p_tr,
			p_val=p_val,
			py1_y0=py1_y0,
			py1_y0_s=py1_y0_s,
			pflip0=pflip0,
			pflip1=pflip1,
			clean_back=clean_back,
			random_seed=random_seed)

	# --load splits
	train_data, valid_data, shifted_data_dict = load_created_data(
		experiment_directory=experiment_directory, py1_y0_s=py1_y0_s)
	# --this helps auto-set training steps at train time
	training_data_size = len(train_data)

	# Build an iterator over training batches.
	def train_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
		dataset = dataset.shuffle(int(1e5)).batch(batch_size).repeat(num_epochs)
		# dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label_given_pixel,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
		return valid_dataset

	# -- Create kfold splits
	if Kfolds > 0:
		effective_validation_size = int(int(len(valid_data) / Kfolds) * Kfolds)
		batch_size = int(effective_validation_size / Kfolds)

		valid_splits = np.random.choice(len(valid_data),
			size=effective_validation_size, replace=False).tolist()

		valid_splits = [
			valid_splits[i:i + batch_size] for i in range(0, effective_validation_size,
				batch_size)
		]

		def Kfold_input_fn_creater(foldid):
			fold_examples = valid_splits[foldid]
			valid_fold_data = [
				valid_data[i] for i in range(len(valid_data)) if i in fold_examples
			]

			def Kfold_input_fn(params):
				map_to_image_label_given_pixel = functools.partial(map_to_image_label,
					pixel=params['pixel'])
				valid_dataset = tf.data.Dataset.from_tensor_slices(valid_fold_data)
				valid_dataset = valid_dataset.map(map_to_image_label_given_pixel)
				valid_dataset = valid_dataset.batch(len(valid_fold_data))
				return valid_dataset
			return Kfold_input_fn
	else:
		Kfold_input_fn_creater = None

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'])
		shifted_test_data = shifted_data_dict[py]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, Kfold_input_fn_creater, eval_input_fn_creater
