# Reproducability Challenge: Learning to Count Everything
Reproducibility project for EECS 6322 by Andrew Luther and Shahak Rozenblat.
Code based on the paper "Learning to Count Everything" by Ranjan et al.

## Requirements

We managed our packages using pip/virtualenv with python version 3.11.5.

To create a virtual environment on Windows, use:

```
python3 -m venv .venv
source .venv/bin/activate
```

To create a virtual environment on MacOS, use:
```
python -m venv .venv
.venv\Scripts\activate
```

Then, install dependencies in your virtual environment with:
```
pip install -r requirements.txt
```

## Dataset Download

To install the dataset, go to:
https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing 

Unzip this folder, and place the unzipped folder (called FSC147_384_V2) directly inside of the "data" directory of the project.

## Training

To train the model on the entire dataset, run the following commands:
```
cd src
python train.py -e=<number of epochs> -b=<number of batches>
```

Training on the entire dataset will automatically save a model to the "saved_models" folder with the name \<Current DateTime\>.pth

To train the model on just the first data point (as we did in our debugging) run the following commands:
```
cd src
python train.py --single -e=<number of epochs> -b=<number of batches>
```

For more information about commandline arguments, run the following command:
```
python train.py --help
```

## Evaluation

