# FineTuning
A collection of fine tuned models and applications based on a broad range of datasets. Repository structure follows this general convention:

1. app

    1. Dataset 1
        * Application and dataset specific scripts and files are stored here.
    2. Dataset 2
        * Application and dataset specific scripts and files are stored here.
    3. etc...

2. datasets

    1. Dataset 1
        * Dataset 1 repository stored here. 
        * Dataset 1 related scripts should automatically download/unzip dataset into this folder.
    2. Dataset 2
        * Dataset 2 repository stored here.
        * Dataset 2 related scripts should automatically download/unzip dataset into this folder.
    3. etc...

3. docs

    1. Dataset 1
        * Any documentation or relevant information for any aspect of this dataset.
    2. Dataset 2
        * Any documentation or relevant information for any aspect of this dataset. 
    3. etc...

4. models

    1. Dataset 1
        * Any number of models that were fine tuned based on Dataset 1.
        * Naming convention is {dataset name}_{base model name}.pt.
    2. Dataset 2
        * Any number of models that were fine tuned based on Dataset 2.
        * Naming convention is {dataset name}_{base model name}.pt.
    3. etc...

5. scripts

    1. Dataset 1
        * All scripts related to model training.
        * Naming convention is {dataset name}_{base model name}.py.
    2. Dataset 2
        * All scripts related to model training.
        * Naming convention is {dataset name}_{base model name}.py.
    3. etc...