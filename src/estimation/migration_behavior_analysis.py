"""
迁移行为分析模块：从迁移历史中提取特征并定义类型
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from scipy.stats import entropy


def extract_migration_history_features(observed_data: pd.DataFrame) -> pd.DataFrame:
    """
    从迁移历史中提取特征，用于类型识别
    
    Args:
        observed_data: 观测数据
        
    Returns:
        DataFrame: 包含迁移特征的DataFrame
    """
    # 按个体和年份分组并排序
    data_sorted = observed_data.sort_values(['individual_id', 'year_t'])
    
    # 初始化特征字典
    features_list = []
    
    for individual_id, group in data_sorted.groupby('individual_id'):
        # 计算个人的详细迁移特征
        individual_features = calculate_individual_migration_features(group)
        individual_features['individual_id'] = individual_id
        features_list.append(individual_features)
    
    features_df = pd.DataFrame(features_list)
    
    return features_df


def calculate_individual_migration_features(individual_data: pd.DataFrame) -> Dict[str, float]:
    """
    计算个体迁移特征
    """
    features = {}
    
    # 确定位置列名称
    location_col = 'provcd_t'  # 根据数据结构，使用省份代码作为位置
    time_col = 'year_t'  # 使用年份作为时间
    
    # 1. 迁移频率特征
    # 计算位置变化次数作为迁移次数
    locations = individual_data[location_col].values
    n_moves = sum(1 for i in range(1, len(locations)) if locations[i] != locations[i-1]) 
    total_periods = len(individual_data)
    features['migration_rate'] = n_moves / max(total_periods - 1, 1) if total_periods > 1 else 0
    features['migration_count'] = n_moves
    
    # 2. 迁移目的地多样性
    unique_destinations = individual_data[location_col].nunique()
    features['exploration_index'] = unique_destinations / max(total_periods, 1)
    
    # 3. 回流迁移特征（返回之前居住过的地方）
    locations_seq = individual_data[location_col].values
    return_count = 0
    visited_locations = set()
    for i, loc in enumerate(locations_seq):
        if i > 0 and loc in visited_locations:
            return_count += 1
        visited_locations.add(loc)
    features['return_migration_ratio'] = return_count / max(n_moves, 1) if n_moves > 0 else 0
    
    # 4. 年龄相关特征
    ages = individual_data['age'].values if 'age' in individual_data.columns else individual_data['age_t'].values
    if len(ages) > 0:
        features['avg_age'] = np.mean(ages)
        features['age_std'] = np.std(ages)
        
        # 迁移时平均年龄（只对发生迁移的时期）
        move_ages = []
        for i in range(1, len(locations_seq)):
            if locations_seq[i] != locations_seq[i-1]:  # 发生了迁移
                move_ages.append(ages[i])
        if move_ages:
            features['avg_migration_age'] = np.mean(move_ages)
        else:
            features['avg_migration_age'] = features['avg_age']
    else:
        features['avg_age'] = 0
        features['age_std'] = 0
        features['avg_migration_age'] = 0
    
    # 5. 居住持续性特征
    stay_durations = []
    if len(locations) > 0:
        current_location = locations[0]
        current_duration = 1
        for i in range(1, len(locations)):
            if locations[i] == current_location:
                current_duration += 1
            else:
                stay_durations.append(current_duration)
                current_location = locations[i]
                current_duration = 1
        stay_durations.append(current_duration)  # 添加最后一段停留时间
    
    if stay_durations:
        features['avg_stay_duration'] = np.mean(stay_durations)
        features['stay_duration_std'] = np.std(stay_durations)
    else:
        features['avg_stay_duration'] = 1
        features['stay_duration_std'] = 0
    
    # 6. 位置变化频率
    features['location_change_frequency'] = n_moves / max(total_periods - 1, 1) if total_periods > 1 else 0
    
    return features


def identify_migration_behavior_types(observed_data: pd.DataFrame, n_types: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于迁移行为特征识别类型
    现在增加了更明确的类型分离初始化
    
    Args:
        observed_data: 观测数据
        n_types: 类型数量
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (类型分配, 后验概率)
    """
    # 提取迁移特征
    features_df = extract_migration_history_features(observed_data)
    
    # 选择用于聚类的特征列
    feature_cols = [
        'migration_rate', 
        'migration_count',
        'exploration_index', 
        'return_migration_ratio', 
        'avg_age',
        'avg_migration_age',
        'avg_stay_duration',
        'location_change_frequency'
    ]
    
    # 清理数据并选择存在的列
    available_feature_cols = [col for col in feature_cols if col in features_df.columns]
    feature_data = features_df[available_feature_cols].fillna(0)
    
    # 如果数据不足以支撑聚类，则使用启发式方法
    if len(feature_data) < n_types:
        # 如果个体数少于类型数，使用均匀分布
        n_individuals = len(observed_data['individual_id'].unique())
        type_assignments = np.random.choice(n_types, size=n_individuals)
        type_probs_matrix = np.full((n_individuals, n_types), 1.0/n_types)
        return type_assignments, type_probs_matrix
    
    # 如果数据不足以支撑GMM，则使用K-means
    if len(feature_data) < 3 * n_types:  # GMM通常需要相对较多的样本
        from sklearn.cluster import KMeans
        # 使用KMeans进行硬聚类，然后转换为概率
        kmeans = KMeans(n_clusters=n_types, random_state=42, n_init=10)
        type_assignments = kmeans.fit_predict(feature_data)
        
        # 将硬分配转换为软概率，增加随机性以避免完全退化
        n_individuals = len(type_assignments)
        type_probs_matrix = np.zeros((n_individuals, n_types))
        
        for i, assigned_type in enumerate(type_assignments):
            # 给分配的类型高概率，其他类型低概率但非零
            type_probs_matrix[i, assigned_type] = 0.8
            for j in range(n_types):
                if j != assigned_type:
                    type_probs_matrix[i, j] = 0.2 / (n_types - 1)
    else:
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # 使用Gaussian Mixture Model进行软聚类
        gmm = SklearnGaussianMixture(n_components=n_types, random_state=42, covariance_type='full')
        type_assignments = gmm.fit_predict(scaled_features)  # 硬聚类
        type_posteriors = gmm.predict_proba(scaled_features)  # 软聚类概率
        
        # 确保没有类型概率为0，防止EM算法中的退化
        type_posteriors = np.maximum(type_posteriors, 0.05)  # 设置最小概率
        type_posteriors = type_posteriors / type_posteriors.sum(axis=1, keepdims=True)  # 重新归一化
        
        # 将结果映射回原始个体ID顺序
        individual_ids = observed_data['individual_id'].unique()
        id_to_idx = {id_: idx for idx, id_ in enumerate(individual_ids)}
        
        n_individuals = len(individual_ids)
        type_probs_matrix = np.zeros((n_individuals, n_types))
        
        features_individual_ids = features_df['individual_id'].values
        id_to_feature_idx = {id_: idx for idx, id_ in enumerate(features_individual_ids)}
        
        for idx, individual_id in enumerate(individual_ids):
            if individual_id in id_to_feature_idx:
                feature_idx = id_to_feature_idx[individual_id]
                if feature_idx < len(type_posteriors):
                    type_probs_matrix[idx, :] = type_posteriors[feature_idx, :]
                else:
                    # 使用均匀分布
                    type_probs_matrix[idx, :] = 1.0 / n_types
            else:
                # 使用均匀分布
                type_probs_matrix[idx, :] = 1.0 / n_types
    
    return type_assignments, type_probs_matrix


def create_behavior_based_initial_params(n_types: int = 3) -> Dict[str, Any]:
    """
    基于行为模式创建初始参数
    
    Args:
        n_types: 类型数量
        
    Returns:
        Dict[str, Any]: 初始参数字典
    """
    initial_params = {
        "alpha_w": 1.0, 
        "lambda": 2.0, 
        "alpha_home": 1.0,
        "rho_base_tier_1": 1.0, 
        "rho_edu": 0.1, 
        "rho_health": 0.1, 
        "rho_house": 0.1,
        "gamma_1": -0.1, 
        "gamma_2": 0.2,
        "gamma_3": -0.4, 
        "gamma_4": 0.01, 
        "gamma_5": -0.05,
        "alpha_climate": 0.1, 
        "alpha_health": 0.1, 
        "alpha_education": 0.1, 
        "alpha_public_services": 0.1,
        "n_choices": 31
    }
    
    # 为每种类型设置不同类型特定参数
    for t in range(n_types):
        # 根据典型行为模式设置参数
        if t == 0:  # 机会型：迁移频繁，距离远
            initial_params[f'gamma_0_type_{t}'] = 0.1  # 低迁移成本
            initial_params[f'gamma_1_type_{t}'] = -0.5  # 低距离敏感性
            initial_params[f'alpha_home_type_{t}'] = 0.1  # 低家乡溢价
            initial_params[f'lambda_type_{t}'] = 2.5  # 高损失厌恶
        elif t == 1:  # 稳定型：迁移很少
            initial_params[f'gamma_0_type_{t}'] = 5.0  # 高迁移成本
            initial_params[f'gamma_1_type_{t}'] = -3.0  # 高距离敏感性
            initial_params[f'alpha_home_type_{t}'] = 2.0  # 高家乡溢价
            initial_params[f'lambda_type_{t}'] = 1.2  # 低损失厌恶
        else:  # 适应型：中等行为
            initial_params[f'gamma_0_type_{t}'] = 1.5  # 中等迁移成本
            initial_params[f'gamma_1_type_{t}'] = -1.5  # 中等距离敏感性
            initial_params[f'alpha_home_type_{t}'] = 0.8  # 中等家乡溢价
            initial_params[f'lambda_type_{t}'] = 1.8  # 中等损失厌恶
    
    return initial_params


def classify_individual_types(observed_data: pd.DataFrame, type_posteriors: np.ndarray) -> Dict[str, Any]:
    """
    基于后验概率分类个体类型
    
    Args:
        observed_data: 观测数据
        type_posteriors: 类型后验概率矩阵
        
    Returns:
        Dict包含类型分类信息
    """
    n_individuals = len(observed_data['individual_id'].unique())
    n_types = type_posteriors.shape[1]
    
    # 为每个个体分配最可能的类型
    assigned_types = np.argmax(type_posteriors, axis=1)
    
    # 计算类型分布
    type_counts = np.bincount(assigned_types, minlength=n_types)
    type_probs = type_counts / len(assigned_types)
    
    # 计算置信度（最大后验概率的平均值）
    confidences = np.max(type_posteriors, axis=1)
    avg_confidence = np.mean(confidences)
    
    return {
        'assigned_types': assigned_types,
        'type_probabilities': type_probs,
        'individual_confidences': confidences,
        'avg_confidence': avg_confidence,
        'n_individuals': n_individuals
    }


if __name__ == "__main__":
    # 示例用法
    # 这里可以添加测试代码
    pass