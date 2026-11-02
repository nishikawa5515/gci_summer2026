import pandas as pd
import numpy as np
import os

# 1. パスの設定
folder = './submissions'
file_optuna = os.path.join(folder, 'submission_optuna.csv')
file_seed_avg = os.path.join(folder, 'submission_seed_avg.csv')
file_final_24 = os.path.join(folder, 'submission_08803.csv')      # 👑 エース（LB: 0.89678ベース）
file_kmeans = os.path.join(folder, 'submission_kmeans.csv')       # 🌶️ 新しいスパイス

print("4つの提出ファイルを読み込んでいます...")
sub_optuna = pd.read_csv(file_optuna)
sub_seed_avg = pd.read_csv(file_seed_avg)
sub_final_24 = pd.read_csv(file_final_24)
sub_kmeans = pd.read_csv(file_kmeans)

# 2. 予測値の抽出
p_optuna = sub_optuna['target'].values
p_seed_avg = sub_seed_avg['target'].values
p_final_24 = sub_final_24['target'].values
p_kmeans = sub_kmeans['target'].values

# ==========================================
# 【新・黄金比率のブレンド】
# エース（final_24）の重み60%は維持しつつ、
# OptunaとSeedAvgを少し削り、KMeansを隠し味（10%）として加えます。
# ==========================================
w_optuna = 0.15
w_seed_avg = 0.15
w_final_24 = 0.60
w_kmeans = 0.10

print(f"\nブレンドを実行中...")
print(f"比率: Optuna={w_optuna} / SeedAvg={w_seed_avg} / Final24={w_final_24} / KMeans={w_kmeans}")

blended_preds = (p_optuna * w_optuna) + \
                (p_seed_avg * w_seed_avg) + \
                (p_final_24 * w_final_24) + \
                (p_kmeans * w_kmeans)

# 3. 新しい提出ファイルの作成
submission = pd.DataFrame({
    'customer_id': sub_final_24['customer_id'],
    'target': blended_preds
})

output_filename = os.path.join(folder, 'submission_blend_v2_kmeans.csv')
submission.to_csv(output_filename, index=False)

print(f"\n✨ 究極の4種ブレンドファイル '{output_filename}' を作成しました！ ✨")
