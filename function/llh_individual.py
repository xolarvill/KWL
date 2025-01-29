import numpy as np
import sympy as sp



def create_llh_individual(individual_index, dataframe, geodata):
    """
    Calculate the likelihood of an individual history (including wages and migration decisions)
    
    Input parameters:
    ----------
    dataframe (pandas.DataFrame): dataset about migration
    geodata (pandas.DataFrame): dataset about location
    individual_index (int): specified index of the individual
    
    Returns
    -------
    llh_commitment (Matrix): 
    """
    # 定义符号参数
    alpha0, alphaK, alphaH, beta1, beta2, gamma0_tau, gamma1, H = sp.symbols(
        'alpha0 alphaK alphaH beta1 beta2 gamma0_tau gamma1 H'
    )
    nu = sp.symbols('nu_1 nu_2 nu_3')  # 地区匹配效应支持点
    eta = sp.symbols('eta_1 eta_2 eta_3')  # 个体固定效应支持点
    beta = sp.symbols('beta')  # 折现因子
    sig_eps = sp.symbols('sig_eps')  # 暂态效应标准差
    
    # 提取个体数据
    individual_data = dataframe[dataframe['pid'] == individual_index].sort_values('year')
    T = len(individual_data)
    ages = individual_data['age'].tolist()
    locations = individual_data['provcd'].tolist()
    incomes = individual_data['income'].tolist()
    
    # 存储符号表达式
    rho_vector = sp.Matrix.zeros(T, 1)
    wage_likelihood_vector = sp.Matrix.zeros(T, 1)
    
    # 反向值迭代初始化
    bar_v_next = 0  # 最后一期的未来效用为0
    
    # 遍历每一期（反向迭代）
    for t in reversed(range(T)):
        current_age = ages[t]
        current_loc = locations[t]
        prev_loc = locations[t-1] if t > 0 else None
        
        # 获取地理信息
        current_geo = geodata[geodata['provcd'] == current_loc].iloc[0]
        mu = current_geo['mean_wage']  # 地区平均收入
        
        # 年龄效应
        G_a = beta1 * current_age + beta2 * current_age**2
        
        # 遍历所有可能的地点计算效用
        v_j_list = []
        wage_likelihoods = []
        
        for j in geodata['provcd'].unique():
            j_geo = geodata[geodata['provcd'] == j].iloc[0]
            j_mu = j_geo['mean_wage']
            j_amenity = j_geo['amenity']
            
            # 是否搬家
            is_move = sp.Integer(1) if j != current_loc else sp.Integer(0)
            
            # 搬家成本
            delta = gamma0_tau + gamma1 * 0  # 假设其他因素n=0
            
            # 货币收入（使用nu的期望）
            w_j = j_mu + nu[0] + G_a + eta[0]  # 示例使用第一个支持点
            
            # 基础效用
            u_j = alpha0 * w_j + alphaK * j_amenity + alphaH * H * (1 if j == 'home' else 0) - delta * is_move
            
            # 未来效用期望
            if j == current_loc:
                future_util = beta * bar_v_next
            elif j == prev_loc and prev_loc is not None:
                future_util = beta * bar_v_next
            else:
                # 新地点，计算期望（简化处理）
                future_util = beta * (sum(nu)/3 + sum(eta)/3)
                
            v_j = u_j + future_util
            v_j_list.append(v_j)
            
            # 工资似然（正态分布）
            if t < T-1 and j == locations[t+1]:
                epsilon = incomes[t] - (mu + nu[0] + eta[0] + G_a)
                z = epsilon / sig_eps
                wage_likelihood = sp.pdf(sp.Normal('z', 0, 1))(z) / sig_eps
                wage_likelihoods.append(wage_likelihood)
        
        # 计算bar_v(x)
        sum_exp_v = sum(sp.exp(vj) for vj in v_j_list)
        bar_v = sp.log(sum_exp_v)
        
        # 计算rho(x,j)
        actual_j = locations[t+1] if t < T-1 else current_loc
        j_index = list(geodata['provcd'].unique()).index(actual_j)
        v_j_actual = v_j_list[j_index]
        rho = sp.exp(v_j_actual - bar_v)
        
        # 存储到向量
        rho_vector[t] = rho
        if wage_likelihoods:
            wage_likelihood_vector[t] = wage_likelihoods[0]
    
    llh_commitment = np.dot(np.array(rho_vector).T, np.array(wage_likelihood_vector))
    
    return llh_commitment

# 示例调用
# dataframe = pd.DataFrame(...)  # 包含个体数据
# geodata = pd.DataFrame(...)  # 包含地理数据
# rho_vec, wage_vec = generate_likelihood_vectors(1, dataframe, geodata)