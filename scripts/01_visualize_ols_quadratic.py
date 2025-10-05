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

# 7. 在完整数据集上训练模型
print("\n训练模型...")
ols_pipeline.fit(X, y)
lgbm_pipeline.fit(X, y)

# 检查 OLS 模型的系数
print(f"\nOLS 模型系数分析:")
regressor = ols_pipeline.named_steps['regressor']
print(f"Age 系数: {regressor.coef_[0]:.6f}")
print(f"Age2 系数: {regressor.coef_[1]:.6f}")
print(f"截距: {regressor.intercept_:.6f}")

# 创建一个简化版本，仅关注年龄和年龄平方
# 创建一个包含平均分类特征值的数据框
age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
age2_range = age_range ** 2

# 创建一个只包含年龄特征的简单模型来更好地展示非线性关系
# 我们将固定分类变量到平均值（对于独热编码，这表示最常见类别）
X_simple = pd.DataFrame({
    'age': age_range,
    'age2': age2_range
})

# 需要处理分类变量，我们使用训练数据中的平均值或最常见类别
# 为了简化，我们构建一个完整特征向量用于预测
X_full_for_prediction = X.iloc[:100]  # 取前100行，但替换age和age2
for i in range(min(100, len(X_full_for_prediction))):
    X_full_for_prediction.iloc[i, 0] = age_range[i]  # age
    X_full_for_prediction.iloc[i, 1] = age2_range[i]  # age2

# 预测不同年龄下的收入
predictions = ols_pipeline.predict(X_full_for_prediction)

# 重新可视化以清楚地显示模型学到的非线性关系
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 对于 OLS 模型，我们直接用年龄和年龄平方的系数来展示关系
# 为了更直观地显示 OLS 模型中年龄和年龄平方的非线性关系，我们创建一个单独的可视化
ax[0].plot(age_range, predictions, 'b-', label='OLS Fitted Curve', linewidth=2)
ax[0].set_title("OLS Model Fitted Relationship: Age vs Log Income")
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Predicted Log Income")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

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
pdp_path = 'results/ml_comparison/age_partial_dependence_plot_corrected.png'
plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
print(f"修正后的PDP图片已保存到 {pdp_path}")

plt.show()

# 生成额外的可视化，单独展示 OLS 模型中的二次关系
plt.figure(figsize=(10, 6))
# 从训练数据中提取 OLS 模型的年龄相关系数
age_coef = regressor.coef_[0]
age2_coef = regressor.coef_[1]
intercept = regressor.intercept_

# 生成预测值，仅考虑年龄和年龄平方的影响（保持其他特征在平均状态）
ols_predictions = intercept + age_coef * age_range + age2_coef * age2_range
plt.plot(age_range, ols_predictions, 'r-', label='OLS Quadratic Curve (Age + Age²)', linewidth=2)
plt.title("OLS Model: Quadratic Relationship between Age and Log Income\n(Age Coef: {:.6f}, Age2 Coef: {:.6f})".format(age_coef, age2_coef))
plt.xlabel("Age")
plt.ylabel("Predicted Log Income (holding other features constant)")
plt.grid(True, alpha=0.3)
plt.legend()

# 保存这个单独的 OLS 二次关系图
ols_quadratic_path = 'results/ml_comparison/ols_quadratic_relationship.png'
plt.savefig(ols_quadratic_path, dpi=300, bbox_inches='tight')
print(f"OLS 二次关系图已保存到 {ols_quadratic_path}")
plt.show()

print("\n脚本执行完成!")