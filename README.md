# GP_regression

## Overview
ガウス過程回帰関連のコードをアップロードします。

## Environment setup
numpy・scipy・matplotlib が必要です。

## Scripts
### Sample_prior_from_GP.py
1 次元と 2 次元の入力空間において、ガウス過程の事前分布から関数をサンプリングします。  
関数 kernel_function の引数 kernel_func で、使用したいカーネル関数を指定します。  

### GP_regression.py
真のデータに対してノイズが乗った学習データを用意し、ガウス過程回帰を行います。  
カーネル関数には RBF kernel + White kernel をデフォルトで用いますが、その他のカーネル関数も使用可能です。  
カーネルのハイパーパラメータは固定値を使用します（最尤推定によるパラメータ最適化は行いません）。

### Bayesian_optimization.py
GP_regression.py と同様にガウス過程回帰を行った後、テストデータの各入力における獲得関数の値を計算します。  
また、獲得関数が最大となる入力を可視化します。  
獲得関数には EI 関数をデフォルトで用いますが、PI 関数や UCB 関数も使用可能です。

## Todo
variance が負になった場合の処理の追加

