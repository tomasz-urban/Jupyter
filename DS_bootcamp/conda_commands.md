# Anaconda useful commands

## We can create environment by giving it a name or specyfing exact path for the environment. 
## In the second case we create a kind of package (for the project) ready for sharing with other people.
## To work on this env (run env/jupyter) we have to be in the exact folder (path)

### Checking already installed environments:
conda info --envs

### Creating environment (v1):
conda create --name ... python=3 #put name of env instead of dots
## Activating environment (v1):
conda activate ... #put name of env instead of dots
## Installing kernel (You can change between kernels in jupyter)
pip install ipykernel

### Creating environment (v2):
1. Create directory with the project name.
2. In this directory run:
conda create --prefix ./env pandas numpy matplotlib scikit-learn

... and You add here all the libraries that You want to install
### Activating environment (v2):
conda activate ... (in here You add full path to the env folder)

If You forget to install some package it can be installed in the activated env
   (for example in here jupyter notebook should be installed from the very beginning):

conda install jupyter 



