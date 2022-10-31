# Introduction
The associated machine learning code for the paper "Learning from Demonstration using a Curvature Regularized
Variational Auto-Encoder (CurvVAE)" by Travers Rhodes, Tapomayukh Bhattacharjee, and Daniel D. Lee.


# How this code can be used:
## Data loading
To download the dataset we use from
```
@incollection{DVN/8TTXZ7/IDCWI2_2018,
author = {Bhattacharjee, Tapomayukh and Song, Hanjun and Lee, Gilwoo and Srinivasa, Siddhartha S.},
publisher = {Harvard Dataverse},
title = {subject10_banana_wrenches_poses.tar.gz},
booktitle = {A Dataset of Food Manipulation Strategies},
year = {2018},
version = {V15},
doi = {10.7910/DVN/8TTXZ7/IDCWI2},
url = {https://doi.org/10.7910/DVN/8TTXZ7/IDCWI2}
}
```
you can run:
```
cd data
source download_fork_trajectory_data.sh
```

This will download the data recordings to the /data folder (for many types of food, not just banana).

## Data cleaning
To clean the data, run the notebook `/notebooks/(01) Code to Clean Fork Pickup Data.ipynb`
changing the variable `foodname = "carrot"` to `"banana"` or other food item.

This will generate a set of files like `/notebooks/banana_clean_pickups/pickup_attempt0.npy` containing individual attempts to pick up food items, with outlier motions removed.

## Model Training
To train a CurvVAE model on the cleaned trajectory data, run the notebook `notebooks/(02) Train Pickup Model (BetaVAE or CurvVAE).ipynb`
changing the variable `foodname = "carrot"` to `"banana"` or other food item.

This will train and save a model to a file named something like `notebooks/trainedmodels/banana_lat3_curvreg0.001_beta0.001_20220209-120436`

Likewise, you can train a PCA model using `notebooks/(03) Train Pickup Model (PCA).ipynb`


## Model application
To create an HTTP service that can apply models, you can run
```
cd applymodelserver
python3 apply_passthrough_model_server.py
```
after modifying `apply_passthrough_model_server.py` to set `loaded_vae = lm.load_model(f"../notebooks/trainedmodels/banana_lat3_curvreg0.001_beta0.001_20220209-120436","cpu")` to point to the name of the model you want to apply.
