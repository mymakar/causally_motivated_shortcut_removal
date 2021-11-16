# Instructions to run the chexpert experiments

## Step 0: Download the chexpert dataset
Follow the [instructions](https://stanfordmlgroup.github.io/competitions/chexpert/) to download the Chexpert dataset. We used the CheXpert-v1.0 Original dataset (\~439G).

## Step 1: Create the cohort for the study
To do that, run
```bash
python -m chexpert.cohort_creation \
--chexpert_directory /path/to/downloaded_chexpert \
--save_directory /path/to/where_to_save_file
```
*Note*: the code assumes that `/path/to/where_to_save_file` will have a pickle file called config.pkl that has all the parameters of the model. This is relavent for cross validation.

## Step 2:
In `chexpert/data_builder.py` modify the `DATA_DIR` to the path where you saved the created cohort from step 1, and modify `MAIN_DIR` to be the path where the chexpert data is downloaded.


## Step 3:
To train a single model, run the following command after replacing `/path/to/where_to_save_model` and `gpuid` with appropriate values.

```bash
python -m chexpert.main --alpha 10.0 --architecture pretrained_densenet \
	--balanced_weights True --batch_size 16 --dropout_rate 0.0 --embedding_dim -1 \
	--l2_penalty 0.0 --minimize_logits False --num_epochs 10 --pixel 512 --random_seed 0 \
	--sigma 10.0 --skew_train True --weighted_mmd True \
	--exp_dir /path/to/where_to_save_model --gpuid gpuid
```
This will run the model with alpha (i.e., the MMD penalty) = 10.0, sigma (i.e., MMD kernel) = 10.0, using a weighted_mmd (i.e., using our weighting scheme described in the paper).

## Step 4:
To cross-validate, run
```bash
python -m shared.cross_validation \
	--results_file /path/to/csv_with_all_results.csv \
	--all_model_dir '/path/to/saved_model_1,/path_to_saved_model_2' \
	--hparams 'embedding_dim,alpha,sigma,l2_penalty,dropout_rate' \
	--weighted_xv 'weighted_bal'  \
	--dataset  'chexpert'\
	--data_dir /path/to/place_where_experiment_data_is/ \
	--pval 0.05 \
	--kfolds  3
```