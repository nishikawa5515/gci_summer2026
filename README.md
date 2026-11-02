# GCI 2026 Summer 受講記録（提出物・自作コード）

東京大学のデータサイエンス講座「GCI 2026 Summer」を受講し、**Advanced を修了**しました。
このリポジトリは、そのうち公開できるもの――各回の提出スクリプトと、講座内データ分析コンペに自分で取り組んだコード・提出物――をまとめたものです。

Pythonの基礎から、科学計算（NumPy）、データ加工（pandas）、データ可視化（Matplotlib）、統計・確率、教師なし学習までを学び、あわせてデータ分析コンペで提出と改善を繰り返しました。

## 公開範囲について

講義ノートブック、教材（PDF・スライド）、講座から配布されたデータセット、解答ノートブックは、講座のルールに従って公開していません。これらは非公開の保管用リポジトリ等で管理しています。
このリポジトリに含まれているのは、本人が作成した提出スクリプト・コンペ用コード・コンペ提出ファイルのみです。

## フォルダ構成

| フォルダ | 内容 |
|---|---|
| `2kaime` | 第2回 Python基礎の提出スクリプト（`homework.py`） |
| `3kaime` | 第3回 Pythonによる科学計算（NumPy）の提出スクリプト（`homework.py`） |
| `4kaime` | 第4回 Pythonによるデータ加工処理の基礎（pandas）の提出スクリプト（`homework.py`） |
| `5kaime` | 第5回 Pythonによるデータ可視化の基礎（Matplotlib）、統計・確率の提出スクリプト（`submit.py`） |
| `6kaime` | 第6回 提出用スクリプト（`submit.py`） |
| `9kaime` | 第9回 教師なし学習の自作スクリプト（`a.py`） |
| `1_kompe` | データ分析コンペ 課題1の作業フォルダ（自作コードと提出ファイル） |
| `2_kompe` | データ分析コンペ 課題2の作業フォルダ（下記参照） |

※ 第7回・第8回については提出スクリプトの形での成果物が残っていないため、本リポジトリには含まれていません。

## コンペでの取り組み（`2_kompe`）

コンペ課題2では、予測モデルの改善を繰り返しました。`submissions` フォルダには提出ファイルが14件残っており、ファイル名からも手法の変遷が追えます。

- 単体モデル: LightGBM（`submission_lgbm_v2.csv` など）、CatBoost（`submission_catboost.csv`）
- 工夫の追加: 特徴量の追加（`submission_lgbm_feat.csv`）、Optunaによる調整（`submission_optuna.csv`）、シード平均（`submission_seed_avg.csv`）、ターゲットエンコーディング（`submission_te.csv`）、疑似ラベリング（`submission_pseudo_label.csv`）、KMeansの特徴量とLightGBMの組み合わせ（`kmeans_lgbm.py`）
- アンサンブル: 複数モデルの予測を混ぜるブレンディング（`blend_submissions.py` → `blend_submissions_v2.py`）と、スタッキング（`advanced_stacking.py`、`submission_stacking.csv`、`submission_ensemble_v3.csv`、`submission_ultra_blend.csv`）

課題1（`1_kompe`）でも、ベースラインを改良した自作コード（`main_improved.py`、`evaluate_improved.py`）と提出ファイルが残っています。

2026年6月にコンペを開始してから7月に終えるまで繰り返し提出を行い、提出スコア0.88の記録もあります。一度の提出で終えず、結果を見て手法を変えながら改善を続ける進め方を実践しました。

## 学んだこと

- Pythonによるデータ分析の一連の流れ（データの読み込み・加工・可視化・モデリング・評価）
- NumPy / pandas / Matplotlib を使った科学計算とデータ処理の基礎
- 統計・確率の基礎と、教師なし学習（クラスタリングなど）の考え方
- コンペを通じた、特徴量の工夫やアンサンブルなど「モデルをどう改善するか」の試行錯誤

## 環境

- Pythonのバージョンは `.python-version` で指定しています
- 講義・宿題のノートブック自体は公開範囲外のため含んでいません（上記「公開範囲について」参照）
