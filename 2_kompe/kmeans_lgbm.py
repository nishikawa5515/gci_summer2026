import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 2. 【NEW】KMeansクラスタリングによるペルソナ作成
# 購買行動（各カテゴリの消費額）をベースにグループ分けします
spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
all_data[spend_cols] = all_data[spend_cols].fillna(0)

# KMeansは金額の大小に引きずられやすいため、事前に標準化（スケーリング）します
scaler = StandardScaler()
scaled_spends = scaler.fit_transform(all_data[spend_cols])

# 顧客を5つのグループに分類
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
all_data['cluster_persona'] = kmeans.fit_predict(scaled_spends)

# 分類結果をTrainとTestに割り当て
train['cluster_persona'] = all_data.iloc[:len(train)]['cluster_persona'].values
test['cluster_persona'] = all_data.iloc[len(train):]['cluster_persona'].values

# 3. 特徴量作成（V1ベース + クラスタ情報）
def create_features(df):
    df = df.copy()
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['age'] = 2026 - df['birth_year']

    df['total_spend'] = df[spend_cols].sum(axis=1)

    purchase_cols = ['web_purchases', 'catalog_purchases', 'store_purchases']
    df['total_purchases'] = df[purchase_cols].sum(axis=1)

    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['is_married'] = df['marital_status'].apply(lambda x: 1 if x in ['Married', 'Together'] else 0)
    df['family_size'] = 1 + df['is_married'] + df['total_children']

    df = df.drop(columns=['registration_date'])
    df = pd.get_dummies(df, columns=['education_level', 'marital_status'], drop_first=True)

    # クラスタをカテゴリ変数として扱うために型変換
    df['cluster_persona'] = df['cluster_persona'].astype('category')
    return df

print("KMeansで顧客ペルソナを作成中...")
train_processed = create_features(train)
test_processed = create_features(test)

train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)
test_processed = test_processed.drop(columns=['target'])

X = train_processed.drop(columns=['customer_id', 'target'])
y = train_processed['target']
X_test = test_processed.drop(columns=['customer_id'])

# 4. 最強の10-Fold CV シードアベレージング
best_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbose': -1,
    'learning_rate': 0.03214991300054569, 'num_leaves': 88, 'max_depth': 5,
    'feature_fraction': 0.8076714735823799, 'bagging_fraction': 0.6070547458788909,
    'bagging_freq': 3, 'min_child_samples': 30,
}

SEEDS = [42, 77, 2026]
FOLDS = 10
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"\n--- {FOLDS}分割交差検証 × {len(SEEDS)}シード の学習を開始 ---")

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    params = best_params.copy()
    params['random_state'] = seed

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_va, y_va, reference=lgb_train)

        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_valid],
            num_boost_round=1500, callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        valid_pred = model.predict(X_va)
        oof_preds[valid_idx] += valid_pred / len(SEEDS)
        test_preds += model.predict(X_test) / (FOLDS * len(SEEDS))

final_auc = roc_auc_score(y, oof_preds)
print("\n" + "="*40)
print(f"★★★ KMeans追加モデル CV Score (AUC): {final_auc:.4f} ★★★")
print("="*40)

submission = pd.DataFrame({'customer_id': test['customer_id'], 'target': test_preds})
# 後でブレンドしやすいように submissions フォルダへ直接保存します
submission.to_csv('./submissions/submission_kmeans.csv', index=False)
print("\n提出ファイル './submissions/submission_kmeans.csv' を作成しました！")
