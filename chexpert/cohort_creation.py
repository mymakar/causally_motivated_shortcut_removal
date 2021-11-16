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
"""Script to create the chexpert dataset"""
import argparse
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

def main(chexpert_directory, save_directory):
	# combine the train/validation data into one big dataset
	trdf = pd.read_csv(f'{chexpert_directory}/train.csv')
	vdf = pd.read_csv(f'{chexpert_directory}/valid.csv')
	df = trdf.append(vdf)
	del trdf, vdf
	# only keep healthy patients and penumonia patients
	df = df[((df['No Finding'] == 1) | (df['Pneumonia'] == 1))]

	# create a unique ID for each patient/study
	df['patient'] = df.Path.str.extract(r'(patient)(\d+)')[1]
	df['study'] = df.Path.str.extract(r'(study)(\d+)')[1].astype(int)
	df['uid'] = df['patient'] + "_" + df['study'].astype(str)
	df = df[['uid', 'patient', 'study', 'Sex', 'Frontal/Lateral', 'Pneumonia', 'Path']]

	# get the main outcome
	df['y0'] = df['Pneumonia'].copy()
	df.y0.fillna(0, inplace = True)
	df.y0[(df.y0 == -1)] = 1
	df.y0.value_counts(dropna = False, normalize = True)

	# get the auxiliary label
	df = df[(df.Sex != 'Unknown')]
	df['y1'] = (df.Sex == 'Male').astype(int)
	df.drop('Sex', axis = 1, inplace = True)

	# keep only studies with frontal views
	# PS: almost all have fontal views (only 0.019% don't)
	df['frontal'] = (df['Frontal/Lateral'] == 'Frontal').astype(int)
	df = df[(df.frontal ==1)]


	# some final cleanups
	df.drop_duplicates(subset=['uid'], inplace = True)
	df.drop(['Frontal/Lateral', 'frontal', 'Pneumonia'], axis = 1, inplace = True)

	# save file
	df.to_csv(f'{save_directory}/clean_data.csv', index = False)


if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--chexpert_directory', '-chexpert_directory',
		help="Directory where the chexpert data is saved",
		type=str)

	parser.add_argument('--save_directory', '-save_directory',
		help="Directory where the final cohort will be saved",
		type=str)

	args = vars(parser.parse_args())
	main(**args)
