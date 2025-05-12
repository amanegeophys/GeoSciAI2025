# GeoSciAI2025 – 地震波形デノイジング

## 概要

本リポジトリは、GeoSciAI2025 地震課題における「地震波形の信号対雑音比（SNR）を向上させる深層学習モデルの開発」に取り組んだ際に作成したものです。  
主に日本の地震観測網 MeSO-net により取得された波形データを対象に、従来の DeepDenoiser モデルの出力をクリーン信号とした U-net ベースのデノイジングモデルを開発・評価しています。  
MSE に加え、SNR と CC を加味した複合的な損失関数としています。

## ディレクトリ構成

```
GeoSciAI2025/
├── model/
│   ├── best_model.pt         # 学習済みの最良モデル
│   ├── CBAM_loss_results.pt  # 評価関数の結果
│   └── loss_log.csv          # 学習推移ログ
├── preprocessing2.ipynb       # データ読み込み・前処理（npz形式対応）
├── test_evaluation.ipynb      # SNR, CC 評価指標の計算（公式スクリプト）
├── train_unet.py              # ベースモデルの学習スクリプト
└── figures                    # 作成された画像の一部
```

## 使用データ

- **データ形式**: `.npz`（各ファイルに3成分波形 + P/S波到達時刻を含む）
- **サンプリング周波数**: 100 Hz（元データは200 Hzからダウンサンプリング）
- **出典**: 東京大学地震研究所「首都圏観測地震波形データセット」  
  https://www.eri.u-tokyo.ac.jp/project/iSeisBayes/dataset/

## 実行手順

1. **データの前処理**
    - `preprocessing2.ipynb` を実行し、波形データを読み込み・前処理します。
2. **モデルの学習**
    - 以下のコマンドで学習を開始します:
      ```bash
      python train_unet.py
      ```
3. **評価**
    - `test_evaluation.ipynb` を実行して、SNR・CC の評価指標を算出し、CSV形式で出力します。

## 評価指標

- **SNR (Signal-to-Noise Ratio)**：信号対雑音比。クリーン信号との比較によって性能を評価。
- **CC (Cross-Correlation)**：デノイズ前後の波形の相互相関係数。信号の形状維持度を示す。

## 参考

- Woollam et al. (2022), *DeepDenoiser: A Deep Learning Seismic Denoising Tool*, https://doi.org/10.1785/0220210324
- GeoSciAI2025, https://sites.google.com/jpgu.org/geosciai2025