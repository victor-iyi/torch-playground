<!--
 Copyright 2022 Victor I. Afolabi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
# PyTorch Playground

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/victor-iyi/torch-playground/main.svg)](https://results.pre-commit.ci/latest/github/victor-iyi/torch-playground/main)

> This repository is developed for educational purposes using the official
[PyTorch tutorials][tutorials].

[tutorials]: https://pytorch.org/tutorials/index.html

**NOTE** Due to issues installing PyTorch with `poetry`, I've commented out the
dependency in `pyproject.toml` and installed it with the following command.
Visit the [PyTorch homepage][pytorch] to check install options with GPU.

```sh
pip install torch torchvision torchaudio
```

[pytorch]: https://pytorch.org

## Introduction to PyTorch

- [x] Quickstart
- [ ] Tensors
- [ ] Datasets & DataLoaders
- [ ] Transforms
- [ ] Build the Neural Network
- [ ] Automatic Differentiation with `torch.autograd`
- [ ] Optimizing Model Parameters
- [ ] Save and Load the Model

## Learning PyTorch

- [ ] Deep Learning with PyTorch: A 60 Minute Blitz
- [ ] Learning PyTorch with Examples
- [ ] What is `torch.nn` really?
- [ ] Visualizing Models, Data and Training with TensorBoard

## Image and Video

- [ ] TorchVision Object Detection Finetuning Tutorial
- [ ] Transfer Learning for Computer Vision Tutorial
- [ ] Adversarial Example Generation
- [ ] DCGAN Tutorial
- [ ] Spatial Transformer Networks Tutorial
- [ ] Optimizing Vision Transformer Model for Deployment

## Audio

- [ ] Audio I/O
- [ ] Audio Resampling
- [ ] Audio Data Augmentation
- [ ] Audio Feature Augmentation
- [ ] Audio Datasets
- [ ] Speech Recognition with Wav2Vec2
- [ ] Speech Command Classification with `torchaudio`
- [ ] Forced Alighment with Wav2Vec2

## Text

- [ ] Language Modeling with `nn.Transformer` and `TorchText`
- [ ] Fast Transformer Inference with Better Transformer
- [ ] NLP From Scratch: Classifying Names with a Character-Level RNN
- [ ] NLP From Scratch: Generating Names with a Character-Level RNN
- [ ] NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
- [ ] Text classification with `torchtext` library

## Reinforcement Learning

- [ ] Reinforcement Learning (DQN) Tutorial
- [ ] Train a Mario-playing RL Agent

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (Apache)

This project is opened under the [Apache License 2.0][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/torch-playground
[issues]: https://github.com/victor-iyi/torch-playground/issues
[license]: ./LICENSE
