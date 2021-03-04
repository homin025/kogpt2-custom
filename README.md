# KoGPT2 Custom

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [License](#license)
- [Reference](#reference)

## Background

This repository is for training KoGPT-2 with two-versioned methods.

First version, language-modeling version, is to train the model without specific task. Its concept is identical to the concept of pre-training method. Its purpose is teaching model to learn about the data's tone or style (e.g. the tone of the news, the tone of the conversation).

Second version, fine-tuning version, is to train the model to perform specific task. Its purpose is teaching model to do the NLP tasks (e.g. summarization, question generation).

The difference between two versions are in [language-modeling/data.py](https://github.com/homin025/kogpt2-custom/blob/main/language-modeling-version/data.py) and [fine-tuning/data.py](https://github.com/homin025/kogpt2-custom/blob/main/fine-tuning-version/data.py).

## Install

This project uses [pytorch](https://pytorch.org/), [transformers](https://github.com/huggingface/transformers), [tokenizers](https://github.com/huggingface/tokenizers).

```sh
$ pip install -r requirements.txt
```

## Usage

For Training

```sh
python train.py --epoch 100 --batch_size 4 --save ./checkpoint/ --load ./checkpoint/model.ckpt --train_dataset ./dataset/train.json --valid_dataset ./dataset/valid.json
```

For Generating

```sh
python generate.py --load ./checkpoint/model.ckpt --top_k 50 --top_p 1.0 --temperature 1.3 --length 150
```

## License

[MIT](LICENSE) © Minho Lee

## Reference

from [openai/gpt-2](https://github.com/openai/gpt-2), 
[SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2), 
[gyunggyung/KoGPT2-FineTuning](https://github.com/gyunggyung/KoGPT2-FineTuning), 
[taeminlee/KoGPT2-Transformers](https://github.com/taeminlee/KoGPT2-Transformers).
