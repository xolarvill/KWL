import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保可重现性
RANDOM_STATE = 42

# 1. 加载数据
print("加载数据...")
df = pd.read_csv('data/processed/clds_detailed.csv')

# 2. 特征工程
print("进行特征工程...")
# 重命名列以匹配要求的列名
df.rename(columns={
    'age': 'age',
    'income': 'income'
}, inplace=True)

# 确保 'age' 列是浮点类型以避免 PDP 警告
df['age'] = df['age'].astype(float)

# 创建目标变量 log_income
df['log_income'] = np.log(df['income'] + 1)

# 创建 age2（年龄平方）
df['age2'] = df['age']**2

# 选择需要的列
required_columns = ['log_income', 'age', 'age2', 'education', 'marital_status', 'health_status']
df = df[required_columns].dropna()

print(f"数据形状: {df.shape}")
print(f"数据列: {list(df.columns)}")

# 显示分类变量的类别
print(f"\n教育类别: {df['education'].unique()}")
print(f"婚姻状况类别: {df['marital_status'].unique()}")
print(f"健康状况类别: {df['health_status'].unique()}")

# 3. 定义特征列表
numerical_features = ['age', 'age2']
categorical_features = ['education', 'marital_status', 'health_status']

# 4. 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# 5. 定义特征和目标
X = df[numerical_features + categorical_features]
y = df['log_income']

print(f"\nX.shape: {X.shape}, y.shape: {y.shape}")

# 6. 创建模型管道

# OLS 基线模型
ols_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# LightGBM 模型
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        random_state=RANDOM_STATE,
        n_estimators=100,  # 为了加快运行速度，使用较小的值
        learning_rate=0.1,
        max_depth=5,
        verbose=-1
    ))
])

# 7. 交叉验证设置
print("\n开始交叉验证...")

# 8. 评估 OLS 模型
print("评估 OLS 模型...")
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ols_rmse_scores = cross_val_score(ols_pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
ols_r2_scores = cross_val_score(ols_pipeline, X, y, cv=cv, scoring='r2')

# 9. 评估 LightGBM 模型
print("评估 LightGBM 模型...")
lgbm_rmse_scores = cross_val_score(lgbm_pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
lgbm_r2_scores = cross_val_score(lgbm_pipeline, X, y, cv=cv, scoring='r2')

# 10. 计算平均值和标准差
ols_rmse_mean = -ols_rmse_scores.mean()
ols_rmse_std = ols_rmse_scores.std()
ols_r2_mean = ols_r2_scores.mean()
ols_r2_std = ols_r2_scores.std()

lgbm_rmse_mean = -lgbm_rmse_scores.mean()
lgbm_rmse_std = lgbm_rmse_scores.std()
lgbm_r2_mean = lgbm_r2_scores.mean()
lgbm_r2_std = lgbm_r2_scores.std()

# 11. 打印结果表格
print("\n" + "="*60)
print("模型性能比较结果")
print("="*60)
print(f"{'模型':<20} {'RMSE (均值 ± 标准差)':<25} {'R² (均值 ± 标准差)':<25}")
print("-"*60)
print(f"{'OLS':<20} {ols_rmse_mean:.4f} ± {ols_rmse_std:.4f}{'':<5} {ols_r2_mean:.4f} ± {ols_r2_std:.4f}")
print(f"{'LightGBM':<20} {lgbm_rmse_mean:.4f} ± {lgbm_rmse_std:.4f}{'':<5} {lgbm_r2_mean:.4f} ± {lgbm_r2_std:.4f}")
print("="*60)

# 12. 在完整数据集上训练模型并检查 OLS 系数
print("\n在完整数据集上训练模型以生成系数...")
ols_pipeline.fit(X, y)
lgbm_pipeline.fit(X, y)

# 检查 OLS 模型的系数
print(f"\nOLS 模型系数分析:")
preprocessor_fitted = ols_pipeline.named_steps['preprocessor']
regressor = ols_pipeline.named_steps['regressor']
feature_names = (
    numerical_features + 
    list(preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
print(f"特征名称: {feature_names[:10]}...")  # 显示前10个特征
print(f"回归系数: {regressor.coef_[:10]}...")  # 显示前10个系数

# 查找 age 和 age2 的系数
age_idx = None
age2_idx = None
for i, name in enumerate(feature_names):
    if 'age' in name and 'age2' not in name:  # 匹配 'age' 但不包含 'age2'
        age_idx = i
    elif 'age2' in name:
        age2_idx = i

if age_idx is not None:
    print(f"Age 系数: {regressor.coef_[age_idx]:.6f}")
if age2_idx is not None:
    print(f"Age2 系数: {regressor.coef_[age2_idx]:.6f}")
    
print(f"截距: {regressor.intercept_:.6f}")

# 13. 生成 PDP 图
print("生成部分依赖图 (PDP)...")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 为 OLS 模型生成 PDP
PartialDependenceDisplay.from_estimator(
    ols_pipeline, X, ['age'], 
    kind='average',
    ax=ax[0],
    grid_resolution=50  # 减少计算量
)
ax[0].set_title("PDP from OLS Model (Quadratic)")
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Partial Dependence (Log Income)")

# 为 LightGBM 模型生成 PDP
PartialDependenceDisplay.from_estimator(
    lgbm_pipeline, X, ['age'], 
    kind='average',
    ax=ax[1],
    grid_resolution=50  # 减少计算量
)
ax[1].set_title("PDP from LightGBM Model (Non-parametric)")
ax[1].set_xlabel("Age")
ax[1].set_ylabel("Partial Dependence (Log Income)")

plt.tight_layout()

# 保存 PDP 图片到 results/ml_comparison 目录
pdp_path = 'results/ml_comparison/age_partial_dependence_plot.png'
plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
print(f"PDP 图片已保存到 {pdp_path}")

plt.show()

print("\n脚本执行完成!")