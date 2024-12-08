# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=hifigan HYDRA_CONFIG_ARGUMENTS
```

Where `hifigan` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To download weights for HIFI-GAN, run the following command:

```bash
python3 download_weigths.py --model-url GOOGLE_DISK_LINK
```
model_url is optional script, inside script model_url has default value of weights.

To run inference (evaluate the model or save predictions):

```bash
python3 synthesize.py HYDRA_CONFIG_ARGUMENTS
```
In `synthesize.yaml`, config for inference, you should set path for checkpoint of model. If you want synthesize from wavs, you should set synthesize_from_audio in synthesize.yaml in datasets and in synthesize_from_audio.yaml set wavs_dir for your wavs. 

If you want synthesize from texts, you should set synthesize_from_text in synthesize.yaml in datasets and in synthesize_from_text.yaml set text_dir for your texts. Also you can synthesize text from cli. Then you should run next command:
```bash
python3 synthesize.py inferencer.text="your_text"
```

After all types of inference in your terminal you will see MOS score.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
