# MLOps DTU course, January 2022
==============================

A repo for the CNN MNIST classifier for the DTU MLOps course. The repo follows the cookiecutter
template structure given by https://drivendata.github.io/cookiecutter-data-science/.

It implements a neural network toy model to predict images from a corrupted MNIST dataset.
The model is not the focus, rather the structure, and practical aspects of the
project repository.

The following notes are for my own learning and easy recalling of the used frameworks.

## Setting up a Cookiecutter project structure
To use the cookiecutter structure for Python, first `pip install cookiecutter` and 
then run in the terminal: 
`cookiecutter https://drivendata.github.io/cookiecutter-data-science/`
This will fetch the template structure and guide you through a short setup process,
naming the project, its repository, Python version etc.

## Updating requirements.txt file
Use `pip list --format=freeze > requirements.txt` to update the requirements.txt file. 
This has the disadvantage of simply listing every single package installed 
in the environment at the time, rather than just the packages imported 
by the project files. 

To do this, the `pipreqs` package is handy. Use it with:
    
    pipreqs .

However, this doesn't always seem to detect 
all the imported modules in the project files. Moreover, it seems to 
not be able to detect the installed version of each module, as it simply scans the 
project files and not the actual 
environment with the installed modules. Therefore, it simply puts the newest version 
of the found modules, which can cause compatibility issues in some cases. 

## Setup
First install all requirements by running:

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


## Profiling with cProfile and snakeviz
To profile runtime of a Python script with cProfile, use e.g.:
    
    python -m cProfile -s time -o .\src\models\VAE\vae_mnist_working.prof .\src\models\VAE\vae_mnist_working.py

This makes a profile of the script "vae_mnist_working.py" and stores the result as a 
.prof file (for Snakeviz) "vae_mnist_working.prof".

Next, we call Snakeviz to visualize the profiling results in the browser:

    snakeviz src/models/VAE/vae_mnist_working.prof

The profiling column `tottime` is the total time spent in a particular 
function *alone*, whereas `cumtime` is the total time spent in the particular 
function *plus* all functions called by it. 

Comparing the results in `vae_mnist_working.prof` and 
`vae_mnist_working_optimized.prof`, respectively, shows that a small 
optimization yielded 3 times faster run speed overall. Simply 
converting the dataset to `TensorDataset` in the beginning of the script, 
before training the model was enough:

    train_dataset = TensorDataset(train_dataset.data.type(torch.float32) / 255, train_dataset.targets)
    test_dataset = TensorDataset(test_dataset.data.type(torch.float32) / 255, test_dataset.targets)

## Monitoring with Weights and Biases
Model training can be monitored using the framework called Weights 
and Biases: https://wandb.ai/hviidhenrik. This is implemented in the VAE example 
"vae_mnist_working.py". Specifically, at the end of each training epoch, it logs: 
   - average batch loss over the entire epoch
   - visualization of some test input digits for the VAE
   - visualization of the reconstructions based on the input digits
   - visualization of generated samples using Gaussian noise fed through 
the trained decoder

## Unit tests
Testing code is important. The Pytest framework is useful for this and is installed
with pip:

    pip install pytest 

A folder called `tests` in the project root should then contain test 
files with all unit tests. These are then run by invoking pytest:

    pytest tests/

### Coverage
How much of the code is covered by the unit tests can be assessed using coverage
reports. Run:

    pip install coverage

And then:

    coverage run -m pytest tests/
    coverage report -m

To get a percentage overview of the number of lines covered by the tests. 
The `-m` flag simply appends the line numbers that are NOT covered. 