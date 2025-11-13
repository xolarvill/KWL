"""
统一管理模型的所有配置参数和初始值
"""
from enum import StrEnum
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np


class OutputLanguage(StrEnum):
    """输出语言格式"""
    LATEX = 'LaTeX'
    MARKDOWN = 'Markdown'
    HTML = 'HTML'


class OutputFileFormat(StrEnum):
    """输出文件格式"""
    LATEX = '.tex'
    MARKDOWN = '.md'
    PLAINTEXT = '.txt'


class OutputStyle(StrEnum):
    """输出表格样式"""
    BOOKTAB = "Booktab"
    PLAIN = "Plain"
    GRID = "Grid"


@dataclass
class ModelConfig:
    """
    模型配置类

    功能：
    1. 管理所有数据文件路径
    2. 管理所有超参数和结构参数的初始值
    3. 提供统一的参数访问接口
    """

    # ========================
    # 一、数据路径参数
    # ========================

    processed_data_dir: str = 'data/processed'
    individual_data_path: str = 'data/processed/clds.csv'
    regional_data_path: str = 'data/processed/geo.xlsx'  # 原始地区数据
    regional_amenity_path: str = 'data/processed/geo_amenities.csv'  # PCA综合amenity指标
    prov_code_ranked_path: str = 'data/processed/prov_code_ranked.json'
    prov_name_ranked_path: str = 'data/processed/prov_name_ranked.json'
    adjacency_matrix_path: str = 'data/processed/adjacent_matrix.xlsx'
    distance_matrix_path: str = 'data/processed/distance_matrix.xlsx'
    linguistic_matrix_path: str = 'data/processed/linguistic_matrix.csv'
    prov_language_data_path: str = 'data/processed/prov_language_data.csv'
    linguistic_data_path: str = 'data/processed/linguistic_tree.json'
    prov_standard_path: str = 'data/processed/prov_standard_map.csv'

    # ========================
    # 二、输出参数
    # ========================

    output_language: OutputLanguage = OutputLanguage.LATEX
    output_file: OutputFileFormat = OutputFileFormat.LATEX
    output_style: OutputStyle = OutputStyle.BOOKTAB
    base_output_dir: str = 'results'
    logs_dir: str = 'progress/log'
    outputs_dir: str = 'results/tables'
    ml_models_dir: str = 'results/ml_models'

    # ========================
    # 三、算法超参数
    # ========================

    ## 优化参数
    max_iter: int = 1000
    tolerance: float = 1e-4

    ## EM算法参数
    em_max_iterations: int = 100
    em_tolerance: float = 1e-4
    em_n_types: int = 3  # 混合模型的类型数量
    
    # 策略模式下的EM容差参数
    fast_em_tolerance: float = 1e-3
    aggressive_em_tolerance: float = 1e-5

    ## M-step中的L-BFGS-B参数
    lbfgsb_maxiter: int = 15
    lbfgsb_gtol: float = 1e-5
    lbfgsb_ftol: float = 1e-6
    
    ## 策略模式配置
    # fast模式配置
    fast_lbfgsb_maxiter: int = 10
    fast_lbfgsb_gtol: float = 1e-3
    fast_lbfgsb_ftol: float = 1e-3
    fast_em_tolerance: float = 1e-3
    fast_bootstrap_em_tol: float = 1e-2
    
    # test模式配置（用于快速测试，采用宽松到夸张的容差）
    test_lbfgsb_maxiter: int = 5
    test_lbfgsb_gtol: float = 1e-2
    test_lbfgsb_ftol: float = 1e-2
    test_em_tolerance: float = 1e-2
    test_bootstrap_em_tol: float = 1e-1

    ## Bootstrap推断参数
    bootstrap_n_replications: int = 200
    bootstrap_max_em_iter: int = 5
    bootstrap_em_tol: float = 1e-3
    bootstrap_seed: int = 42
    bootstrap_n_jobs: int = -1  # -1表示使用所有CPU核心
    
    # 策略模式下的Bootstrap参数
    fast_bootstrap_em_tol: float = 1e-2
    aggressive_bootstrap_em_tol: float = 1e-4

    # ========================
    # 四、模型外生参数
    # ========================

    discount_factor: float = 0.95  # β: 贴现因子
    age_min: int = 18  # 最小年龄
    age_max: int = 70  # 最大年龄
    n_choices: int = 31  # 选择数量（31个省份）

    # ========================
    # 五、结构参数初始值
    # ========================

    # 5.1 共享参数（type-invariant parameters）
    # 这些参数在所有类型间共享

    ## 收入效用
    alpha_w: float = 1.0  # 收入的边际效用

    ## 户籍惩罚参数（三档城市分类）
    rho_base_tier_1: float = 1.0  # 一线城市（北上广深等）的户籍惩罚基础值
    rho_base_tier_2: float = 0.5  # 二线城市的户籍惩罚基础值
    rho_base_tier_3: float = 0.2  # 三线及以下城市的户籍惩罚基础值
    rho_edu: float = 0.1  # 户籍×教育交互项
    rho_health: float = 0.1  # 户籍×医疗交互项
    rho_house: float = 0.1  # 户籍×住房交互项

    ## 地区舒适度参数
    alpha_climate: float = 0.1  # 气候舒适度
    alpha_education: float = 0.1  # 教育舒适度
    alpha_health: float = 0.1  # 医疗舒适度
    alpha_public_services: float = 0.1  # 公共服务舒适度
    alpha_hazard: float = 0.1  # 自然灾害（负面amenity，alpha_hazard > 0意味着灾害越多效用越低）

    ## 迁移成本参数（共享部分）
    gamma_1: float = -0.1  # 距离对迁移成本的影响（注：gamma_0是type-specific）
    gamma_2: float = 0.2  # 邻近性对迁移成本的影响
    gamma_3: float = -0.4  # 回流迁移对迁移成本的影响
    gamma_4: float = 0.01  # 年龄对迁移成本的影响
    gamma_5: float = -0.05  # 人口规模对迁移成本的影响

    ## 共享参数（删除前景理论后简化）
    alpha_home: float = 1.0  # 家乡溢价（默认值）
    # lambda_default 参数已删除（前景理论已移除）

    # 5.2 类型特定参数（type-specific parameters）
    # 这些参数在不同类型间有差异

    ## Type 0: 机会型（Opportunistic）
    # 特征：迁移频繁，低固定迁移成本
    gamma_0_type_0: float = 0.1  # 低固定迁移成本（type-specific）

    ## Type 1: 稳定型（Stable）
    # 特征：迁移很少，高固定迁移成本
    gamma_0_type_1: float = 5.0  # 高固定迁移成本（type-specific）

    ## Type 2: 适应型（Adaptive）
    # 特征：中等迁移频率
    gamma_0_type_2: float = 1.5  # 中等固定迁移成本（type-specific）

    # 5.3 类型概率初始值
    pi_type_0: float = 0.33  # Type 0的先验概率
    pi_type_1: float = 0.33  # Type 1的先验概率
    pi_type_2: float = 0.34  # Type 2的先验概率（加起来=1）

    # ========================
    # 五（续）、离散支撑点配置（论文984-1016行）
    # ========================

    ## 支撑点数量
    n_eta_support: int = 7  # 个体固定效应η_i的支撑点数量
    n_nu_support: int = 5   # 个体-地区收入匹配ν_ij的支撑点数量
    n_xi_support: int = 5   # 个体-地区偏好匹配ξ_ij的支撑点数量
    n_sigma_support: int = 4  # 工资波动性σ_ε的支撑点数量

    ## 支撑点取值范围
    eta_range: Tuple[float, float] = (-2.0, 2.0)  # η_i范围
    nu_range: Tuple[float, float] = (-1.5, 1.5)   # ν_ij范围
    xi_range: Tuple[float, float] = (-1.0, 1.0)   # ξ_ij范围
    sigma_range: Tuple[float, float] = (0.3, 1.5)  # σ_ε范围

    ## ω枚举控制
    max_omega_per_individual: int = 100  # 每个个体的最大ω组合数（超过则用蒙特卡洛）
    use_simplified_omega: bool = True     # 是否使用简化策略（只在访问过的地区实例化ν和ξ）

    ## 互联网效应参数（论文737-742行）
    delta_0: float = 0.5  # 方差基准参数
    delta_1: float = 0.1  # 互联网效应参数（预期>0，即互联网降低不确定性）

    ## 参照工资类型（已废弃 - 前景理论已删除）
    reference_wage_type: str = 'lagged'  # 保留配置但不再使用

    ## 工资似然开关
    include_wage_likelihood: bool = True  # 是否在似然函数中包含工资密度

    ## EM算法控制开关
    use_discrete_support: bool = True  # 是否使用离散支撑点ω（P0.2功能）
    # 设为False可回退到原始EM算法（不使用支撑点枚举）
    
    ## 性能优化参数
    use_jit_compilation: bool = True  # 是否使用JIT编译优化
    parallel_jobs: int = 4  # 并行计算的核心数
    weight_threshold: float = 1e-6  # 权重阈值，用于过滤小权重计算

    # ========================
    # 五（续）、参数边界约束（用于L-BFGS-B优化）
    # ========================

    ## 效用参数边界
    alpha_bounds: Tuple[float, float] = (-2.0, 10.0)  # alpha_* 参数的边界

    ## 户籍惩罚参数边界
    rho_bounds: Tuple[float, float] = (0.0, 10.0)  # rho_* 参数的边界（必须非负）

    ## 迁移成本参数边界
    gamma_0_bounds: Tuple[float, float] = (0.0, 20.0)  # 固定迁移成本（必须非负）
    gamma_1_bounds: Tuple[float, float] = (-5.0, 1.0)  # 距离系数（通常为负）
    gamma_2_bounds: Tuple[float, float] = (-1.0, 5.0)  # 邻近性系数
    gamma_3_bounds: Tuple[float, float] = (-5.0, 1.0)  # 回流迁移系数（通常为负）
    gamma_4_bounds: Tuple[float, float] = (-0.5, 0.5)  # 年龄系数
    gamma_5_bounds: Tuple[float, float] = (-1.0, 1.0)  # 人口规模系数

    ## 其他参数边界
    # lambda_bounds 已删除（前景理论已移除）
    sigma_epsilon_bounds: Tuple[float, float] = (0.1, 5.0)  # 工资波动性（必须为正）

    ## 默认边界（用于未明确指定的参数）
    default_bounds: Tuple[float, float] = (-10.0, 10.0)

    # ========================
    # 六、辅助方法
    # ========================

    def get_initial_params(self, use_type_specific: bool = True) -> Dict[str, Any]:
        """
        获取模型初始参数字典

        Args:
            use_type_specific: 是否使用类型特定参数

        Returns:
            Dict[str, Any]: 参数字典
        """
        params = {
            # 共享参数（所有类型共用）
            "alpha_w": self.alpha_w,
            "rho_base_tier_1": self.rho_base_tier_1,
            "rho_base_tier_2": self.rho_base_tier_2,
            "rho_base_tier_3": self.rho_base_tier_3,
            "rho_edu": self.rho_edu,
            "rho_health": self.rho_health,
            "rho_house": self.rho_house,
            "alpha_climate": self.alpha_climate,
            "alpha_education": self.alpha_education,
            "alpha_health": self.alpha_health,
            "alpha_public_services": self.alpha_public_services,
            "alpha_hazard": self.alpha_hazard,
            "gamma_1": self.gamma_1,
            "gamma_2": self.gamma_2,
            "gamma_3": self.gamma_3,
            "gamma_4": self.gamma_4,
            "gamma_5": self.gamma_5,
            "alpha_home": self.alpha_home,
            # "lambda" 参数已删除（前景理论已移除）
            "n_choices": self.n_choices
        }
        # 注意：gamma_1现在是共享参数

        if use_type_specific:
            # 类型特定参数：只保留gamma_0作为type-specific
            # **参数归一化**: 将 type 0 作为基准组，其固定迁移成本 gamma_0 为0
            params[f"gamma_0_type_0"] = 0.0

            for t in range(1, self.em_n_types): # 从 1 开始循环，跳过基准组
                params[f"gamma_0_type_{t}"] = getattr(self, f"gamma_0_type_{t}")
        else:
            # 非混合模型：使用默认参数
            params["gamma_0"] = self.gamma_0_type_0  # 使用type_0作为默认

        return params

    def get_initial_type_probabilities(self) -> np.ndarray:
        """
        获取类型概率的初始值

        Returns:
            np.ndarray: 类型概率向量 (K,)
        """
        return np.array([self.pi_type_0, self.pi_type_1, self.pi_type_2])

    def get_em_config(self) -> Dict[str, Any]:
        """
        获取EM算法配置

        Returns:
            Dict[str, Any]: EM算法配置字典
        """
        return {
            "max_iterations": self.em_max_iterations,
            "tolerance": self.em_tolerance,
            "n_types": self.em_n_types,
            "beta": self.discount_factor,
            "n_choices": self.n_choices
        }

    def get_bootstrap_config(self) -> Dict[str, Any]:
        """
        获取Bootstrap推断配置

        Returns:
            Dict[str, Any]: Bootstrap配置字典
        """
        return {
            "n_bootstrap": self.bootstrap_n_replications,
            "max_em_iterations": self.bootstrap_max_em_iter,
            "em_tolerance": self.bootstrap_em_tol,
            "seed": self.bootstrap_seed,
            "n_jobs": self.bootstrap_n_jobs
        }

    def get_discrete_support_config(self) -> Dict[str, Any]:
        """
        获取离散支撑点配置

        Returns:
            Dict[str, Any]: 离散支撑点配置字典
        """
        return {
            "n_eta_support": self.n_eta_support,
            "n_nu_support": self.n_nu_support,
            "n_xi_support": self.n_xi_support,
            "n_sigma_support": self.n_sigma_support,
            "eta_range": self.eta_range,
            "nu_range": self.nu_range,
            "xi_range": self.xi_range,
            "sigma_range": self.sigma_range,
            "max_omega_per_individual": self.max_omega_per_individual,
            "use_simplified_omega": self.use_simplified_omega
        }

    def get_parameter_bounds(self, param_names: List[str]) -> List[Tuple[float, float]]:
        """
        根据参数名称列表获取对应的边界约束

        参数:
        ----
        param_names : List[str]
            参数名称列表

        返回:
        ----
        List[Tuple[float, float]]
            每个参数的(下界, 上界)元组列表
        """
        bounds = []
        for name in param_names:
            if name.startswith('alpha_'):
                bounds.append(self.alpha_bounds)
            elif name.startswith('rho_'):
                bounds.append(self.rho_bounds)
            elif name.startswith('gamma_0_type_'):
                bounds.append(self.gamma_0_bounds)
            elif name == 'gamma_1':
                bounds.append(self.gamma_1_bounds)
            elif name == 'gamma_2':
                bounds.append(self.gamma_2_bounds)
            elif name == 'gamma_3':
                bounds.append(self.gamma_3_bounds)
            elif name == 'gamma_4':
                bounds.append(self.gamma_4_bounds)
            elif name == 'gamma_5':
                bounds.append(self.gamma_5_bounds)
            # lambda 参数已删除（前景理论已移除）
            elif name == 'sigma_epsilon':
                bounds.append(self.sigma_epsilon_bounds)
            else:
                bounds.append(self.default_bounds)

        return bounds

    def update_param(self, param_name: str, value: Any) -> None:
        """
        更新单个参数值

        Args:
            param_name: 参数名称
            value: 新值
        """
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise ValueError(f"参数 '{param_name}' 不存在于ModelConfig中")
            
    def get_strategy_params(self, strategy: str = "normal") -> Dict[str, Any]:
        """
        根据策略模式获取相应的参数配置
        
        Args:
            strategy: 策略模式 ("fast", "normal", "test")
            
        Returns:
            Dict[str, Any]: 参数配置字典
        """
        if strategy == "fast":
            return {
                "lbfgsb_maxiter": self.fast_lbfgsb_maxiter,
                "lbfgsb_gtol": self.fast_lbfgsb_gtol,
                "lbfgsb_ftol": self.fast_lbfgsb_ftol,
                "em_tolerance": self.fast_em_tolerance,
                "bootstrap_em_tol": self.fast_bootstrap_em_tol
            }
        elif strategy == "test":
            return {
                "lbfgsb_maxiter": self.test_lbfgsb_maxiter,
                "lbfgsb_gtol": self.test_lbfgsb_gtol,
                "lbfgsb_ftol": self.test_lbfgsb_ftol,
                "em_tolerance": self.test_em_tolerance,
                "bootstrap_em_tol": self.test_bootstrap_em_tol
            }
        else:  # normal mode
            return {
                "lbfgsb_maxiter": self.lbfgsb_maxiter,
                "lbfgsb_gtol": self.lbfgsb_gtol,
                "lbfgsb_ftol": self.lbfgsb_ftol,
                "em_tolerance": self.em_tolerance,
                "bootstrap_em_tol": self.bootstrap_em_tol
            }

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return f"ModelConfig(n_types={self.em_n_types}, n_choices={self.n_choices}, beta={self.discount_factor})"


# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == '__main__':
    config = ModelConfig()
    print("=" * 60)
    print("ModelConfig 配置摘要")
    print("=" * 60)
    print(f"\n1. 模型设置:")
    print(f"   - 类型数量: {config.em_n_types}")
    print(f"   - 选择数量: {config.n_choices}")
    print(f"   - 贴现因子: {config.discount_factor}")

    print(f"\n2. EM算法配置:")
    em_conf = config.get_em_config()
    for k, v in em_conf.items():
        print(f"   - {k}: {v}")

    print(f"\n3. 初始参数（共享参数）:")
    params = config.get_initial_params(use_type_specific=True)
    shared_params = {k: v for k, v in params.items() if 'type_' not in k and k != 'n_choices'}
    for k, v in list(shared_params.items())[:10]:
        print(f"   - {k}: {v}")

    print(f"\n4. 类型特定参数:")
    for t in range(config.em_n_types):
        print(f"   Type {t}:")
        print(f"      gamma_0: {params.get(f'gamma_0_type_{t}', 'N/A')}")

    print(f"\n   共享参数:")
    print(f"      alpha_home: {params['alpha_home']}")
    print(f"      lambda: {params['lambda']}")
    print(f"      gamma_1: {params['gamma_1']}")

    print(f"\n5. Bootstrap配置:")
    bootstrap_conf = config.get_bootstrap_config()
    for k, v in bootstrap_conf.items():
        print(f"   - {k}: {v}")

    print("\n" + "=" * 60)
