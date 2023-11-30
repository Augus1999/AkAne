# A<span style='color:#CB4154'>k</span>Ane: bidirectionary model that predicts molecular properties and generates molecular structures


![OS](https://img.shields.io/badge/OS-Windows%20|%20Linux%20|%20macOS-blue?color=00b166)
![python](https://img.shields.io/badge/Python-3.9%20|%203.10-blue.svg?color=dd9b65)
![torch](https://img.shields.io/badge/torch-2.0-blue?color=708ddd)
![black](https://img.shields.io/badge/code%20style-black-black)

Proudly made in <img src="image/uos_blue.png" alt="University of Southampton" width="100"/>.

<img src="image/model_scheme.png" alt="model scheme" width="600"/>

## Web APP
First download the compiled models (`torchscript_model.7z`) from the [release](https://github.com/Augus1999/AkAne/releases) and extract the folder `torchscript_model` to the same directory of `app.py`. Then you can run `$ python app.py` to launch the web app locally.

A demo is being hosted on [hugging face Spaces](https://huggingface.co/spaces/suenoomozawa/AkAne).

## Trained models
We provide pre-trained autoencoder, prediction models trained on MoleculeNet benchmark (including ESOL, FreeSolv, Lipo, BBBP, BACE, ClinTox, HIV), QM9, PhotoSwitch, AqSolDB, CMC value dataset, and a range of deep eutectic solvents (DES) properties, and 2 generation models that generate protein ligands and DES pairs, respectively.

You can download trained models from the [release](https://github.com/Augus1999/AkAne/releases).

## Dataset format
The datasets we used and provided are stored in CSV files. We provide a python class `CSVData` in [akane2/utils/dataset.py](akane2/utils/dataset.py) to handle these files which require a header with the following tags:
* __smiles__ (_mandatory_): the entities under this tag should be molecule SMILES strings. Multiple tags are acceptable.
* __temperature__ (_optional_): the temperature in kelvin. Providing more than one this tag won't cause any error but only the last one will be accepted.
* __ratio__ (_optional_): molar ratio of each compound in the format of `x1:x2:...:xn`.  Providing more than one this tag won't cause any error but only the last one will be accepted.
* __value__ (_optional_): entities under this tag should be molecular properties. Multiple tags are acceptable and in this case you can tell `CSVData` which value(s) should be loaded by specifying `label_idx=[...]`. If a property is not defined, leave it empty and the entity will be automatically masked to `torch.inf` telling the model that this property is unknown. 
* __seq__ (_optional_): FASTA-style protein sequence. Providing more than one this tag won't cause any error but only the last one will be accepted. NOTE THAT WHEN THIS TAG IS USED, MOLECULAR PROPERTIES (IF PRESENT IN THE FILE) WILL NOT BE LOADED.

These tags are unnecessary to be ordered, e.g.,
```csv
smiles,value,value,ratio,smiles
```
and
```csv
smiles,smiles,ratio,value,value
```
are both okey.

## Training thy own model
The following is a guide of how to train your own model.
#### _1. Create your dataset_
Create your own dataset(s) following the dataset format.
#### _2. Split your dataset_
```python
from akane2.utils import split_dataset

split_ratio = 0.8  # you can use any training:testing ratio from 0 to 1
method = "random"  # another choice is "scaffold"
split_dataset("YOUR_DATASET.csv", split_ratio, method)
```
This will split your dataset into `YOUR_DATASET_train.csv` and `YOUR_DATASET_test.csv`.
#### _3. Load your data_
```python
from akane2.utils import CSVData

limit = None  # you can specify how many data-points your want to load, e.g., 1200
label_index = None  # see the above "Dataset format" section
train_set = CSVData("YOUR_DATASET_train.csv", limit, label_index)
test_set = CSVData("YOUR_DATASET_test.csv", limit, label_index)
```
#### _4. Define your work space_
```python
from pathlib import Path

cwd = Path(__file__).parent
workdir = cwd / "YOUR_WORKDIR"  # the directory where checkpoints (if any) will be stored
logdir = cwd / "YOUR_LOG.log"  # where to print the log (you can set it to "None")
```
#### _5. Define your model_
We provide 2 types of models (that is where _2_ comes from in the package name): `akane2.representation.AkAne` (the whole A<span style='color:#CB4154'>k</span>Ane model) and `akane2.representation.Kamome` (the indenpendent encoder part without latent space regularisation).
* If you are only interested in property predictions or molecule classifications, we recommend to use only the encoder model:
```python
from akane2.representation import Kamome

model = Kamome()  #  DON'T FORGET TO SET THE IMPORTANT HYPERPARAMETERS
```
* If you are going to train a generation or bidirectionary model, please use the whole model:
```python
from akane2.representation import AkAne

model = AkAne()  #  DON'T FORGET TO SET THE IMPORTANT HYPERPARAMETERS
```
__IMPORTANT__: Regarding to the hyperparameters (e.g., `num_task` and `label_mode`) that DEFINE the functionality of the model, please refer to the comments under each model in [representation.py](akane2/representation.py).
#### _6. Train your model_
```python
import os
from akane2.utils import train, find_recent_checkpoint

os.environ["NUM_WORKER"] = "4"  # set `num_workers` of torch.utils.data.DataLoader (the default value is min(4, num_cpu_cores) if you remove this line)
chkpt = find_recent_checkpoint(workdir)  # find latest checkpoint (if any)
mode = "predict"  # training mode based on thy desire. Other options are "autoencoder", "classify", and "diffusion"
n_epochs = 1000  # training epochs
batch_size = 5  # define batch-size. Choose thy own value that won't cause `CUDA out of memory` error
save_every = 100  # save a checkpoint every `save_every` epochs (you can set to "None")
train(model, train_set, mode, n_epochs, batch_size, chkpt, logdir, workdir, save_every)
```
You will find the weight of trained model `trained.pt` and (if any) checkpoint file(s) `state-xxxx.pth` under _workdir_. You can safely delete any checkpoint file if you don't want them.
#### _6. Test your model (ignore this step if you are training an autoencoder or generation model)_
```python
from akane2.utils import test

os.environ["INFERENCE_BATCH_SIZE"] = "20"  # set the inference batch-size that won't cause `CUDA out of memory` error (the default value is 20 if you remove this line)
mode = "prediction"  # testing mode based on thy model. Another choice is "classification"
print(test(model, test_set, mode, workdir/ "train.pt", logdir))
```

## Known issue
You cannot compile 2 or more AkAne models (i.e., `akane2.representation.AkAne`) into TorchScript modules together in one file. We recommend to save the compiled models before hand and load by `torch.jit.load(...)`.