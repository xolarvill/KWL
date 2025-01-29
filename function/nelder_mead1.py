import numpy as np

def nelder_mead(function, start_point, tol=1e-6, max_iter=1000, alpha=1, gamma=2, beta=0.5, delta=0.5):
    """
    Nelder-Mead optimization algorithm.
    
    function: 目标函数，接收一个向量并返回一个标量
    start_point: 初始点，应该是一个包含目标函数维度的向量
    tol: 收敛容忍度
    max_iter: 最大迭代次数
    alpha: 反射系数
    gamma: 扩展系数
    beta: 收缩系数
    delta: 缩小系数
    """
    
    n = len(start_point)  # 目标函数的维度
    simplex = np.zeros((n + 1, n))  # 初始化simplex，n+1个点
    simplex[0] = start_point  # 第一个点为起始点
    
    # 生成其他n个点，偏移初始点
    for i in range(1, n + 1):
        simplex[i] = start_point + np.random.uniform(-1, 1, n)  # 生成随机点
    
    def evaluate_simplex(simplex):
        """计算simplex中每个点的目标函数值"""
        return np.array([function(x) for x in simplex])
    
    def check_convergence(simplex, tol):
        """判断是否收敛"""
        differences = np.linalg.norm(simplex[1:] - simplex[0], axis=1)
        return np.max(differences) < tol
    
    def reflect(simplex, function_values):
        """反射操作"""
        worst = np.argmax(function_values)  # 最差点
        best = np.argmin(function_values)  # 最好点
        reflection = simplex[best] + alpha * (simplex[best] - simplex[worst])
        return reflection
    
    def expand(simplex, function_values, reflection):
        """扩展操作"""
        worst = np.argmax(function_values)
        best = np.argmin(function_values)
        expansion = simplex[best] + gamma * (reflection - simplex[best])
        return expansion
    
    def contract(simplex, function_values, reflection):
        """收缩操作"""
        worst = np.argmax(function_values)
        best = np.argmin(function_values)
        contraction = simplex[best] + beta * (simplex[worst] - simplex[best])
        return contraction
    
    def shrink(simplex):
        """缩小操作"""
        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
        return simplex
    
    # 主要的迭代过程
    for iteration in range(max_iter):
        function_values = evaluate_simplex(simplex)  # 计算目标函数值
        best_index = np.argmin(function_values)
        worst_index = np.argmax(function_values)
        
        # 获取当前最好和最差点
        best_point = simplex[best_index]
        worst_point = simplex[worst_index]
        
        # 如果已经收敛，则结束迭代
        if check_convergence(simplex, tol):
            break
        
        # 反射操作
        reflection = reflect(simplex, function_values)
        reflection_value = function(reflection)
        
        if reflection_value < function_values[best_index]:
            # 如果反射点比最好的点更好，进行扩展
            expansion = expand(simplex, function_values, reflection)
            expansion_value = function(expansion)
            if expansion_value < reflection_value:
                simplex[worst_index] = expansion
            else:
                simplex[worst_index] = reflection
        else:
            # 如果反射点不够好，进行收缩操作
            if reflection_value < function_values[worst_index]:
                simplex[worst_index] = reflection
            else:
                # 如果反射点都不好，进行缩小
                simplex = shrink(simplex)
        
        # 检查是否收敛
        if check_convergence(simplex, tol):
            break
    
    # 返回最优解
    function_values = evaluate_simplex(simplex)
    best_index = np.argmin(function_values)
    return simplex[best_index], function_values[best_index]  # 最优解点及其对应的函数值



# 示例函数
def example_func(x):
    return x[0]**2 + x[1]**2 + 1

x8, x9 = nelder_mead(example_func, np.array([1, 1]))  
print(f"最优解点: {x8}")
print(f"最优解函数值: {x9}")