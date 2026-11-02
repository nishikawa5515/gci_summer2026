import pandas as pd
import numpy as np
import os

# 1. パスの設定（submissionsフォルダの中のファイルを指定）
folder = './submissions'

# 【修正箇所】実際のファイル名に合わせました
file_optuna = os.path.join(folder, 'submission_optuna.csv')
file_seed_avg = os.path.join(folder, 'submission_seed_avg.csv')
file_final_24 = os.path.join(folder, 'submission_08803.csv') # ← ここを修正！

print("提出ファイルを読み込んでいます...")
sub_optuna = pd.read_csv(file_optuna)
sub_seed_avg = pd.read_csv(file_seed_avg)
sub_final_24 = pd.read_csv(file_final_24)

# 2. 予測値の抽出（ターゲット確率）
p_optuna = sub_optuna['target'].values
p_seed_avg = sub_seed_avg['target'].values
p_final_24 = sub_final_24['target'].values

# ==========================================
# 【黄金比率のブレンド】
# 本番で最も強かった「0.8803（本番0.89678）」に高い重みを置きつつ、
# 異なるアプローチで作られた2つのファイルを隠し味として混ぜ合わせます。
# ==========================================
w_optuna = 0.20
w_seed_avg = 0.20
w_final_24 = 0.60

print(f"\nブレンドを実行中（比率: Optuna={w_optuna} / SeedAvg={w_seed_avg} / Final24(08803)={w_final_24}）...")
blended_preds = (p_optuna * w_optuna) + (p_seed_avg * w_seed_avg) + (p_final_24 * w_final_24)

# 3. 新しい提出ファイルの作成
submission = pd.DataFrame({
    'customer_id': sub_final_24['customer_id'],
    'target': blended_preds
})

# 出力ファイル名（これはリネームせずにこのまま出力します）
output_filename = os.path.join(folder, 'submission_ultra_blend.csv')
submission.to_csv(output_filename, index=False)

print(f"\n✨ 究極のブレンドファイル '{output_filename}' を作成しました！ ✨")
print("これをOmnicampusへ提出して、大台の「0.90突破」を狙いに行きましょう！")
