# GeoSciAI2025 – 地震波形デノイジング深層学習モデル

## 概要

本リポジトリは、GeoSciAI2025 地震課題における「地震波形の信号対雑音比（SNR）を向上させる深層学習モデルの開発」に取り組んだ際に作成したものです。  
主に日本の地震観測網 MeSO-net により取得された波形データを対象に、従来の DeepDenoiser モデルの出力をクリーン信号とした U-net ベースのデノイジングモデルを開発・評価しています。
実験的に行った CycleGAN ベースのデノイジングモデルの学習スクリプトも含まれています。

## ディレクトリ構成

```
GeoSciAI2025/
├── model/
│   └── best_model.pt        # 学習済みの最良モデル
├── preprocessing.ipynb      # データ読み込み・前処理（npz形式データ対応）
├── sample_code.ipynb        # SNR, CC 評価関数の算出（公式スクリプト）
├── train_unet.py                 # ベースモデルの学習スクリプト
└── train_cyclegan.py        # CycleGAN による学習スクリプト（実験的）
```

## 使用データ

- **データ形式**: `.npz`（各ファイルに3成分波形 + P/S波到達時刻含む）
- **サンプリング周波数**: 100 Hz（元データは200 Hzからのダウンサンプリング）
- **出典**: 東京大学地震研究所「首都圏観測地震波形データセット」  
  https://www.eri.u-tokyo.ac.jp/project/iSeisBayes/dataset/

## 実行手順

1. **データの前処理**
    - `preprocessing.ipynb` を使用して、データの前処理
2. **モデルの学習**
    - 通常の学習:
      ```bash
      python train_unet.py
      ```
    - CycleGAN による学習（実験的）:
      ```bash
      python train_cyclegan.py
      ```
3. **評価**
    - `sample_code.ipynb` を使用して、SNR・CCを計算し、評価用CSVを出力します。

## 評価指標

- **SNR (Signal-to-Noise Ratio)**：信号対雑音比
- **CC (Cross-Correlation)**：デノイズ前後の波形の相互相関係数

## 引用文献

- Woollam et al. (2022), *DeepDenoiser: A Deep Learning Seismic Denoising Tool*, https://doi.org/10.1785/0220210324