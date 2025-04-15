# Reproducibility Challenge: Learning to Count Everything
Reproducibility project for EECS 6322 by Andrew Luther and Shahak Rozenblat.
The code was written based on the paper "Learning to Count Everything" by Ranjan et al.

## Requirements

We managed our packages using pip/virtualenv with python version 3.11.5.

To create a virtual environment on Windows, use:

```
python -m venv .venv
.venv\Scripts\activate
```

To create a virtual environment on MacOS, use:
```
python3 -m venv .venv
source .venv/bin/activate
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
python train.py -e=<number of epochs> -b=<number of batches> -lr=<learning rate>
```

Training on the entire dataset will automatically save a model to the "saved_models" folder with the name \<Current DateTime\>.pth

To train the model on just the first data point (as we did in our debugging) run the following commands:
```
cd src
python train.py --single -e=<number of epochs> -b=<number of batches> -lr=<learning rate>
```

For more information about commandline arguments, run the following command:
```
python train.py --help
```

Additionally, after training (with any settings), the last prediction that the model makes will be output in the predictions folder, and will be called "final_prediction.png". We also log details about the training in src/runs using tensorboard.

## Model Download

To download a sample trained model, download the .pth file from here: https://drive.google.com/drive/folders/1SePNVieQoO3RK4Fb4RkaFJaciu5ZomWr?usp=drive_link. Then, place the model in the "saved_models" directory.

## Evaluation

To evaluate a trained model, run the following commands:
```
cd src
python eval.py -m=<path to model checkpoint>
```

If you downloaded the sample model as described above (and placed it in the "saved_models" directory), the code will, by default, find that model path without you needing to specify the path explicitly.

Additionally, there are optional parameters to turn on adaptation loss with ```--adaptation```, test on the validation set with ```--validation```, and limit the model to evaluating for the first n data points with ```-l=<limit>```.

Note that, though adaptation loss has been implemented, with our current model the adaptation loss does not improve the model's performance, and simply makes the model take much longer to evaluate (since it trains for 100 iterations on each sample).
