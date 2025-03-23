import numpy as np

def create_sample_likelihood(pi, individual_likelihoods):
    """
    构造样本似然函数，仅存储参数和个体似然函数，不立即计算值。
    
    参数：
        pi (array-like): 类型权重向量，形状为 (K,)
        individual_likelihoods (list of lists): 嵌套列表，形状为 (N, K)，
            其中 individual_likelihoods[i][τ] 是第 i 个个体在类型 τ 下的似然函数。
    
    返回：
        function: 样本似然函数，接受参数 theta（类型参数列表），返回对数似然值。
    """
    # 将 pi 转换为 NumPy 数组以便后续计算
    pi = np.array(pi)
    
    # 定义样本似然函数（闭包）
    def sample_likelihood(theta):
        """
        计算样本对数似然值。
        
        参数：
            theta (list or array-like): 类型参数列表，形状为 (K,)
        
        返回：
            float: 样本对数似然值。
        """
        total = 0.0
        # 遍历所有个体
        for i in range(len(individual_likelihoods)):
            weighted_sum = 0.0
            # 遍历所有类型
            for tau in range(len(pi)):
                # 提取类型 τ 的似然函数和参数 θ_τ
                L_i_tau = individual_likelihoods[i][tau]
                theta_tau = theta[tau]
                # 计算加权似然并累加
                weighted_sum += pi[tau] * L_i_tau(theta_tau)
            # 对每个个体的加权似然取对数并累加到总和
            total += np.log(weighted_sum)
        return total
    
    return sample_likelihood


# 示例使用
if __name__ == '__main__':
    # 个体 0 的类型 0 的似然函数
    def likelihood_i0_tau0(theta_tau):
        return np.exp(-theta_tau**2)

    # 个体 0 的类型 1 的似然函数
    def likelihood_i0_tau1(theta_tau):
        return np.exp(-theta_tau**3)

    # 个体 1 的类型 0 的似然函数
    def likelihood_i1_tau0(theta_tau):
        return np.exp(-theta_tau)

    # 个体 1 的类型 1 的似然函数
    def likelihood_i1_tau1(theta_tau):
        return np.exp(-2 * theta_tau)

    # 组织为 (N, K) 的嵌套列表
    individual_likelihoods = [
        [likelihood_i0_tau0, likelihood_i0_tau1],
        [likelihood_i1_tau0, likelihood_i1_tau1]
    ]

    # 类型权重向量
    pi = [0.4, 0.6]  # 例如：类型 0 占比 40%，类型 1 占比 60%
    
    sample_likelihood = create_sample_likelihood(pi, individual_likelihoods)