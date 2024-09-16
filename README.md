# Passive Inference Attacks on Split Learning via Adversarial Regularization

This is the official implementation of the paper "Passive Inference Attacks on Split Learning via Adversarial Regularization" accepted by NDSS 2025. You can read the full paper [here](https://arxiv.org/pdf/2310.10483).

## Dependencies

To run the experiments in this repo, you need `numpy`, `matplotlib`, `sklearn`, `tensorflow`. On linux, you can install all the dependencies is through conda and pip (please use the CUDA version applicable to your system):

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
cd $CONDA_PREFIX/etc/conda/activate.d/
. ./env_vars.sh

pip3 install numpy==1.26.4
pip3 install matplotlib==3.7.2
pip3 install scikit_learn==1.3.0
pip3 install imageio==2.33.0
pip3 install tensorflow==2.9.0
```

## Datasets setup

In this code base, you can run experiments with `CIFAR-10`, `CIFAR-100`, `Tiny ImageNet`, and `STL-10`. The `CIFAR` datasets are automatically downloaded by `keras` when you run the experiments. The `Tiny ImageNet` and `STL-10` datasets need to be downloaded manually.

To run experiments with dataset Tiny ImageNet, you'll need to first download the dataset and unzip it in the './data' directory:

```
mkdir data
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -qq tiny-imagenet-200.zip
```

To run experiments with dataset STL-10, you'll need to first download the dataset and unzip it in the './data' directory:

```
mkdir data
cd data
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
tar -xvzf stl10_binary.tar.gz
```

## Usage

To run SDAR on CIFAR-10 at level 7 for the 1st trial on vanilla SL with ResNet-20, direct to `scripts` and run:
```
python3 run_sdar.py vsl resnet cifar10 7 1
```

You can also run the experiments with other configurations, the arguments are as follows:
- `vsl` or `ssl` for vanilla SL or U-shaped SL
- `resnet` or `plainnet` for ResNet or PlainNet
- `cifar10`, `cifar100`, `tinyimagenet`, or `stl10` for CIFAR-10, CIFAR-100, Tiny ImageNet, or STL-10
- `4`, `5`, `6`, `7` for the level of the client's side partial model
- any integer for the trial number

Optionally, you can also run the experiments with the following options (all optional and mutually exclusive):
- `--width` for the width of the model with options `standard` (default), `wide`, `narrow`
- `--aux_data_frac` for the fraction of auxiliary data w.r.t. the private target training data, specified with a float number between 0 and 1, default is 1.0
- `--num_class_to_remove` for the number of classes to remove from the auxliary dataset, specified with an integer number, default is 0
- `--diff_simulator`: if specified, the server will use a different architecture for its simulator, only applicable when target model is set to `resnet` where the server's simulator will be set to `plainnet`
- `--ablation` to run experiments with ablation studies, with options `no_e_dis`, `no_d_dis`, `no_dis`, `no_cond`, `no_label_flip`, `naive_sda`
- `--num_hetero_client` for running the attack with multiple heterogeneous clients, specified with an integer number, default is 1

You can also run the defense methods with the following options (all optional and mutually exclusive):
- `--l1`: if specified with a float number as the regularization factor, l1 regularization will be applied in SL training
- `--l2`: if specified with a float number as the regularization factor, l2 regularization will be applied in SL training
- `--dropout`: if specified with a float number as the dropout rate, dropout will be applied in SL training
- `--alpha`: if specified with a float number as the alpha value, the decorrelation defense will be applied in SL training

By default, all results will be stored to the `scripts/{experiment_type}_results` directory, including log, loss history (as `npz` files), and examples of reconstructed images (as `png` files). If you want the log to be directed to `stdout`, you can add `--print_to_stdout` to the command.

Implementation of SDAR can be found in the `src` directory.

## Citation

```bibtex
@inproceedings{10.14722/ndss.2025.23030,
  author       = {Xiaochen Zhu and
                  Xinjian Luo and
                  Yuncheng Wu and
                  Yangfan Jiang and
                  Xiaokui Xiao and
                  Beng Chin Ooi},
  title        = {Passive Inference Attacks on Split Learning via Adversarial Regularization},
  booktitle    = {Network and Distributed System Security (NDSS) Symposium 2025},
  location     = {San Diego, CA},
  year         = {2025},
  month        = {Feburary},
  doi          = {10.14722/ndss.2025.23030},
}
```

## License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2023 Xiaochen Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
