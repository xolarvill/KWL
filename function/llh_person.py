import numpy as np
import pandas as pd
import indicator
from calK import calK
from scipy.stats import norm
from scipy.special import logsumexp
from getsupport import getsupport
from math import exp, log
from itertools import product



def individual_chatgpt(dataframe, geodata, params):
    """
    参数:
    pid: 个体ID
    dataframe: 个体面板数据 (包含pid, year, provcd, income, age)
    geodata: 地区面板数据 (包含provcd, year, mean_wage)
    beta1, beta2: 年龄效应参数
    sigma: 暂态效应标准差
    eta_val: 个体固定效应支持点绝对值
    nu_val: 地区匹配效应支持点绝对值
    
    返回:
    该个体的对数似然值
    """
    df = pd.merge(dataframe, geodata, on="provcd", how="left")
    
    def compute_basic_utility(df, alpha0, alphaK, alphaH, beta1, beta2):
        # 计算基础效用的每个组成部分
        df['u'] = (alpha0 * df['income'] + 
                alphaK * df['amenity'] + 
                alphaH * df['home_effect'] * df['is_home'] - 
                df['moving_cost'] * df['is_moving'])
        
        # 增加年龄效应
        df['G_a'] = beta1 * df['age'] + beta2 * df['age']**2
        df['u'] += df['G_a']
        
        return df

    def compute_expected_utility(df, rho):
        # 计算期望效用 bar_v(x)
        df['v_bar'] = np.log(np.sum(np.exp(df['v'] - df['v'].mean())))
        return df

    def compute_choice_probability(df):
        # 计算选择概率 rho(x,j)
        df['rho'] = np.exp(df['v'] - df['v_bar']) / np.sum(np.exp(df['v'] - df['v_bar']))
        return df

    
    alpha0, alphaK, alphaH, beta1, beta2 = params
        
    # 计算基础效用
    df = compute_basic_utility(df, alpha0, alphaK, alphaH, beta1, beta2)
        
    # 计算期望效用 bar_v(x)
    df = compute_expected_utility(df)
        
    # 计算选择概率 rho(x,j)
    df = compute_choice_probability(df)
        
    # 计算似然函数
    likelihood = 0
    for i in range(len(df)):
        likelihood += np.log(df.loc[i, 'rho'])  # 选择实际地点的概率
            
    return -likelihood  # 返回负对数似然函数，用于最小化


# 获取所有唯一地点和个体信息
all_provcd = ['provcd'].unique()
provcd_map = {provcd: idx for idx, provcd in enumerate(all_provcd)}
J = len(all_provcd)

# 假设折现因子
beta = 0.95

def likelihood_function_deepseek(individual_df, geo_df, nu_support, eta_support, individual_index, J, provcd_map, beta):
    # 预处理数据：先截取个人数据，再合并个体与地理信息，并排序
    individual_data = individual_df[individual_df['pid'] == individual_index][['pid', 'age', 'year', 'income', 'provcd']].sort_values(by='year')
    merged_df = pd.merge(individual_data, geo_df, on='provcd', how='left')
    
    
    # 预处理每个个体的初始状态
    initial_provcd = merged_df[merged_df['year'] == 2010]['provcd'].values[0]
    
    # 定义值函数缓存结构（动态规划用）
    # 假设状态由当前地点、上一个地点、年龄构成，并设定最大年龄
    max_age = merged_df['age'].max()
    value_cache = {}
    



def location_likelihood_function_deepseek_2(individual_data, geo_data, individual_index):
    # 预处理个体数据
    individual_df = individual_data[individual_data['pid'] == individual_index].sort_values('year')
    T = len(individual_df)
    years = individual_df['year'].values
    provcd_sequence = individual_df['provcd'].values
    age_sequence = individual_df['age'].values
    income_sequence = individual_df['income'].values

    # 预处理地理数据
    geo_dict = geo_data.set_index('provcd').to_dict(orient='index')

    # 支持点数目
    num_support_points = 3

    # 定义闭包函数
    def likelihood_function(params):
        alpha0, alphaK, alphaH, beta1, beta2, gamma0_tau = params[:6]
        # 假设eta和nu的支持点参数紧随其后
        support_params = params[6:]
        eta_support = support_params[:num_support_points]
        nu_support = support_params[num_support_points:]

        # 离散概率（均匀分布）
        prob = 1.0 / num_support_points

        # 初始化值函数字典
        v_bar = {}

        # 逆向值迭代
        for t in reversed(range(T)):
            current_provcd = provcd_sequence[t] if t < T else None
            last_provcd = provcd_sequence[t-1] if t > 0 else None
            age = age_sequence[t]

            # 遍历所有可能选择的地点
            for j in geo_dict.keys():
                # 计算基础效用
                G = beta1 * age + beta2 * (age ** 2)
                mu_j = geo_dict[j]['mean_wage']
                K_j = geo_dict[j]['amenity']

                # 计算搬家成本
                if j != current_provcd:
                    move_cost = gamma0_tau
                else:
                    move_cost = 0

                # 遍历所有支持点组合
                total_u = 0.0
                for eta_val, nu_val in product(eta_support, nu_support):
                    w_j = mu_j + nu_val + G + eta_val
                    home_premium = alphaH * (j == current_provcd)  # 假设当前地点为家
                    u = alpha0 * w_j + alphaK * K_j + home_premium - move_cost
                    total_u += u * prob * prob  # 两个独立分布的乘积

                # 计算未来效用（下一时期）
                next_age = age + 1
                next_v_bar = 0.0
                if t + 1 < T:
                    next_provcd = j
                    next_last_provcd = current_provcd
                    next_state = (next_provcd, next_last_provcd, next_age)
                    next_v_bar = v_bar.get(next_state, 0.0)

                # 计算当前v(j)
                v_j = total_u + 0.95 * next_v_bar  # beta=0.95

                # 存储当前状态值
                current_state = (current_provcd, last_provcd, age)
                if current_state not in v_bar:
                    v_bar[current_state] = []
                v_bar[current_state].append(v_j)

            # 计算当前状态的v_bar
            for state in v_bar:
                if len(v_bar[state]) == len(geo_dict):
                    log_sum = log(sum(exp(v) for v in v_bar[state]))
                    v_bar[state] = log_sum

        # 计算每个时期的rho值
        rho_functions = []
        for t in range(T-1):
            current_provcd = provcd_sequence[t]
            last_provcd = provcd_sequence[t-1] if t > 0 else None
            age = age_sequence[t]
            chosen_j = provcd_sequence[t+1]

            def get_rho(t, chosen_j, params):
                current_state = (current_provcd, last_provcd, age)
                v_j = ...  # 根据存储的v_bar计算v_j
                v_bar_val = v_bar.get(current_state, 0.0)
                return exp(v_j - v_bar_val)

            rho_functions.append(lambda p=params: get_rho(t, chosen_j, p))

        return rho_functions
    
    return likelihood_function

# 示例用法
# individual_data = pd.read_csv(...)
# geo_data = pd.read_csv(...)
# likelihood_func = likelihood_function(individual_data, geo_data, 123)
# rho_functions = likelihood_func(params)
# 最终返回的rho_functions包含T-1个函数