# MLOps DTU course, January 2022
==============================

A repo for the CNN MNIST classifier for the DTU MLOps course. The repo follows the cookiecutter
template structure given by https://drivendata.github.io/cookiecutter-data-science/.

It implements a neural network toy model to predict images from a corrupted MNIST dataset.
The model is not the focus, rather the structure, and practical aspects of the
project repository.

## Setup

First install all requirements by running 

    pip install -r requirements.txt

To run the project, do the following:
1. run "make_dataset.py" with input and output parameters - it prepares the data for training:
             
       python src/data/make_dataset.py <input_filepath> <output_filepath>

2. run "train_model.py" which trains the model. Appropriate parameters such as epochs, learning rate, etc
can be given via the command line. A learning curve will be saved to reports/figures as png:

       python src/models/CNN/train_model.py <hyperparameter1> <hyperparameter2> <hyperparameterN>

3. run "predict_model.py" with parameters specifying a serialized model and a data location:

       python src/models/CNN/predict_model.py <model_filepath> <data_filepath>


## Makefile

I did not manage to get the Makefile working. I have some conflicting installation on 
my computer, it seems, that prevents make from working.