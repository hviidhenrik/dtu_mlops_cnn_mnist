# MLOps DTU course, January 2022
==============================

A repo for the CNN MNIST classifier for the DTU MLOps course. The repo follows the cookiecutter
template structure given by https://drivendata.github.io/cookiecutter-data-science/.

It implements a neural network toy model to predict images from a corrupted MNIST dataset.
The model is not the focus, rather the structure, and practical aspects of the
project repository.

## Setup

To run the project, do the following:
1. run "make_dataset.py" with input and output parameters - it prepares the data for training
2. run "train_model.py" which trains the model. Appropriate parameters such as epochs, learning rate, etc
can be given via the command line. A learning curve will be saved to reports/figures as png.
3. run "predict_model.py" with parameters specifying a serialized model and a data location.

## Makefile

I did not manage to get the Makefile working. I have some conflicting installation on 
my computer, it seems, that prevents make from working.