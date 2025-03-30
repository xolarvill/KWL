import llh_log_sample, newton_line_search, nelder_mead, nelder_mead1, compare_vec

# 传统方法：
def traditional():
    # 初始化代估参数
    ## u(x,j)
    alpha0 = 0 # wage income parameter
    alpha1 = 0 # houseprice
    alpha2 = 0 # weather = hazard + temperature + air quality + water supply
    alpha3 = 0 # education 
    alpha4 = 0 # health
    alpha5 = 0 # traffic = public transportation + road service
    alpha6 = 0 # 
    alphaH = 0 # home premium parameter
    xi = 0 # random permanent component
    zeta = 0 # exogenous shock
    ## wage
    nu = 0 # location match effect
    eta = 0 # individual effect
    ## G(a)
    beta1 = 0 # first order age effect
    beta2 = 0 # second order age effect
    ## delta
    gammaF = 0 # mirgration friction parameter
    gamma0 = 0 # heterogenous migration cost parameter
    gamma1 = 0 # moving cost parameter
    gamma2 = 0 # cheaper adjacent location parameter
    gamma3 = 0 # cheaper previous location parameter
    gamma4 = 0 # age-moving cost parameter
    gamma5 = 0 # cheaper larger city parameter 
    ## other
    beta = 0.95 # discount factor
    
    theta = [alpha0, 
             alpha1, alpha2, alpha3, alpha4, alpha5, alpha6,
             alphaH,
             xi, zeta, nu, eta,
             beta1, beta2,
             gammaF,
             gamma0, gamma1, gamma2, gamma3, gamma4, gamma5
             ] # 代估参数向量
    
    # 从个人历史的似然函数得到总的样本似然函数
    pi = 0
    individual_likelihoods = []
    sample_loglikelihood_function = llh_log_sample.create_sample_likelihood(pi, individual_likelihoods)

    # 用经过LU分解优化过的Newton线搜索最大化似然函数（实则是最小化负的似然函数）
    opt_pars, opt_step, num_iterations, se = newton_line_search.newton_line_search(-sample_loglikelihood_function, initial_guess = theta, tolerance = 1e-6, max_iter = 100)
    
    # 记录似然估计参数的标准误
    try:
        print(f"最大化似然函数的最优步长为{opt_step}，共迭代{num_iterations}次")
        with open('std.txt', 'w') as f:
            f.write(f"最大化似然函数的最优步长为{opt_step}，共迭代{num_iterations}次\n")
            for i in range(len(opt_pars)):
                result_str = f"参数{i+1}的似然估计为{opt_pars[i]}，标准误为{se[i]}"
                print(result_str)
                f.write(result_str + '\n')
    except Exception as e:
        error_str = f"参数估计出错: {e}"
        print(error_str)
        with open('std.txt', 'w') as f:
            f.write(error_str + '\n')
    
    # 并用Nelder-Mead算法检查局部最大值（和Newton法一样，原本也是最小化，使目标函数变负即可）
    nm1 = nelder_mead.nelder_mead(-sample_loglikelihood_function)
    nm2 = nelder_mead1.nelder_mead(-sample_loglikelihood_function)
    
    # 比较Newton法和Nelder-Mead法的结果
    # 并给出大致波动范围
    print(f'Newton法与Nelder-Mead法一的比较结果是:{compare_vec.compare_vectors(opt_pars, nm1)}')
    print(f'Newton法与Nelder-Mead法二的比较结果是:{compare_vec.compare_vectors(opt_pars, nm2)}')
    
    # 返回结果        
    print('The algorithm is done')
    print('Read readme.md for more information')