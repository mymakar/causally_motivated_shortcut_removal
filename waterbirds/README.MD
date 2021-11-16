# Instructions to run the waterbirds experiments

## Step 0: Download the datasets
You'll first have to download the [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset including the bird segmentation dataset.
You also need to download the [places](http://places2.csail.mit.edu/download.html) dataset. We used the high-resolution training images (MD5: 67e186b496a84c929568076ed01a8aa1) from Places.

In the paper we explain that the original places dataset has several noisy backgrounds. For example some images included other birds (making the task of bird identification difficult). We picked about 200 clean land backgrounds, and 300 clean water backgrounds and augmented that data by rotations, zooming, and manipulating brightness. A tarball of the clean + augmented background images (\~500M) is [here](https://drive.google.com/file/d/19ZL2WTHOTtEDUSzldZKEb4JqUC-5vX8B/view).
*If you use this tarball, please cite both our paper, and the original places data curators*. The tarball has been shared with the curators' approval.


## Step 1:
In `waterbirds/data_builder.py` modify the `DATA_DIR` to the path where you saved the images from step 0, modify `IMAGE_DIR` to be the path where you downloaded the birds data, and modify the `SEGMENTATION_DIR` to be the path where you've saved the bird segmentations


## Step 2:
To train a single model, run the following command after replacing `/path/to/where_to_save_model` and `gpuid` with appropriate values.

```bash
python -m waterbirds.main --alpha 10.0 --architecture pretrained_resnet \
	--balanced_weights True --batch_size 64 --dropout_rate 0.0 --embedding_dim -1 \
	--l2_penalty 0.0 --minimize_logits False --num_epochs 10 \
	--pflip0 0.01 --pflip1 0.01 --pixel 128 --py0 0.8 --py1_y0 0.9 --random_seed 0 \
	--sigma 10.0  --weighted_mmd True \
	--exp_dir /path/to/where_to_save_model --gpuid gpuid
```
This will run the model with alpha (i.e., the MMD penalty) = 10.0, sigma (i.e., MMD kernel) = 10.0, using a weighted_mmd (i.e., using our weighting scheme described in the paper).

## Step 3:
To cross-validate, run
```bash
python -m shared.cross_validation \
	--results_file /path/to/csv_with_all_results.csv \
	--all_model_dir '/path/to/saved_model_1,/path_to_saved_model_2' \
	--hparams 'embedding_dim,alpha,sigma,l2_penalty,dropout_rate' \
	--weighted_xv 'weighted_bal'  \
	--dataset  'waterbirds'\
	--data_dir /path/to/place_where_experiment_data_is/ \
	--pval 0.05 \
	--kfolds  3
```