# GP_regression

## Overview
Python code about Gaussian process regression.

## Scripts
### Sampling_from_GP.py
1次元, 2次元の入力空間において、ガウス過程の事前分布から関数をサンプリングします。  
関数 kernel_function の引数 kernel_func で、使用したいカーネル関数を指定します。  

### GP_regression.py
真のデータに対してノイズが乗った学習データを用意し、ガウス過程回帰を行います。
カーネル関数は RBF kernel + White kernel を用います。  
カーネルのハイパーパラメータは固定値を使用します（最尤推定によるパラメータ最適化は行いません）。
