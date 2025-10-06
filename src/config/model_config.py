"""
统一管理模型的所有配置参数和初始值
"""
from enum import StrEnum
from typing import List, Dict, Any
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
    regional_data_path: str = 'data/processed/geo.xlsx'
    prov_code_ranked_path: str = 'data/processed/prov_code_ranked.json'
    prov_name_ranked_path: str = 'data/processed/prov_name_ranked.json'
    adjacency_matrix_path: str = 'data/processed/adjacent_matrix.xlsx'
    distance_matrix_path: str = 'data/processed/distance_matrix.xlsx'
    linguistic_matrix_path: str = 'data/processed/linguistic_matrix.csv'
    prov_language_data_path: str = 'data/processed/prov_language_data.csv'
    linguistic_data_path: str = 'data/processed/linguistic_tree.json'

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

    ## M-step中的L-BFGS-B参数
    lbfgsb_maxiter: int = 15
    lbfgsb_gtol: float = 1e-3
    lbfgsb_ftol: float = 1e-3

    ## Bootstrap推断参数
    bootstrap_n_replications: int = 200
    bootstrap_max_em_iter: int = 5
    bootstrap_em_tol: float = 1e-3
    bootstrap_seed: int = 42
    bootstrap_n_jobs: int = -1  # -1表示使用所有CPU核心

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

    ## 户籍惩罚参数
    rho_base_tier_1: float = 1.0  # 第一档地区的户籍惩罚基础值
    rho_edu: float = 0.1  # 户籍×教育交互项
    rho_health: float = 0.1  # 户籍×医疗交互项
    rho_house: float = 0.1  # 户籍×住房交互项

    ## 地区舒适度参数
    alpha_climate: float = 0.1  # 气候舒适度
    alpha_education: float = 0.1  # 教育舒适度
    alpha_health: float = 0.1  # 医疗舒适度
    alpha_public_services: float = 0.1  # 公共服务舒适度

    ## 迁移成本参数（共享部分）
    gamma_1: float = -0.1  # 距离对迁移成本的影响（注：gamma_0是type-specific）
    gamma_2: float = 0.2  # 邻近性对迁移成本的影响
    gamma_3: float = -0.4  # 回流迁移对迁移成本的影响
    gamma_4: float = 0.01  # 年龄对迁移成本的影响
    gamma_5: float = -0.05  # 人口规模对迁移成本的影响

    ## 共享的alpha_home和lambda（如果不做type-specific可以使用）
    alpha_home: float = 1.0  # 家乡溢价（默认值，如果使用type-specific则被覆盖）
    lambda_default: float = 2.0  # 损失厌恶系数（默认值，如果使用type-specific则被覆盖）

    # 5.2 类型特定参数（type-specific parameters）
    # 这些参数在不同类型间有差异

    ## Type 0: 机会型（Opportunistic）
    # 特征：迁移频繁，距离远，对收入机会敏感
    gamma_0_type_0: float = 0.1  # 低固定迁移成本
    gamma_1_type_0: float = -0.5  # 低距离敏感性
    alpha_home_type_0: float = 0.1  # 低家乡溢价
    lambda_type_0: float = 2.5  # 高损失厌恶

    ## Type 1: 稳定型（Stable）
    # 特征：迁移很少，偏好熟悉环境
    gamma_0_type_1: float = 5.0  # 高固定迁移成本
    gamma_1_type_1: float = -3.0  # 高距离敏感性
    alpha_home_type_1: float = 2.0  # 高家乡溢价
    lambda_type_1: float = 1.2  # 低损失厌恶

    ## Type 2: 适应型（Adaptive）
    # 特征：中等迁移频率，平衡收入和家乡偏好
    gamma_0_type_2: float = 1.5  # 中等固定迁移成本
    gamma_1_type_2: float = -1.5  # 中等距离敏感性
    alpha_home_type_2: float = 0.8  # 中等家乡溢价
    lambda_type_2: float = 1.8  # 中等损失厌恶

    # 5.3 类型概率初始值
    pi_type_0: float = 0.33  # Type 0的先验概率
    pi_type_1: float = 0.33  # Type 1的先验概率
    pi_type_2: float = 0.34  # Type 2的先验概率（加起来=1）

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
            # 共享参数
            "alpha_w": self.alpha_w,
            "rho_base_tier_1": self.rho_base_tier_1,
            "rho_edu": self.rho_edu,
            "rho_health": self.rho_health,
            "rho_house": self.rho_house,
            "alpha_climate": self.alpha_climate,
            "alpha_education": self.alpha_education,
            "alpha_health": self.alpha_health,
            "alpha_public_services": self.alpha_public_services,
            "gamma_1": self.gamma_1,
            "gamma_2": self.gamma_2,
            "gamma_3": self.gamma_3,
            "gamma_4": self.gamma_4,
            "gamma_5": self.gamma_5,
            "n_choices": self.n_choices
        }

        if use_type_specific:
            # 类型特定参数
            for t in range(self.em_n_types):
                params[f"gamma_0_type_{t}"] = getattr(self, f"gamma_0_type_{t}")
                params[f"gamma_1_type_{t}"] = getattr(self, f"gamma_1_type_{t}", self.gamma_1)
                params[f"alpha_home_type_{t}"] = getattr(self, f"alpha_home_type_{t}")
                params[f"lambda_type_{t}"] = getattr(self, f"lambda_type_{t}")
        else:
            # 使用默认的alpha_home和lambda
            params["alpha_home"] = self.alpha_home
            params["lambda"] = self.lambda_default
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
        print(f"      gamma_0: {params[f'gamma_0_type_{t}']}")
        print(f"      alpha_home: {params[f'alpha_home_type_{t}']}")
        print(f"      lambda: {params[f'lambda_type_{t}']}")

    print(f"\n5. Bootstrap配置:")
    bootstrap_conf = config.get_bootstrap_config()
    for k, v in bootstrap_conf.items():
        print(f"   - {k}: {v}")

    print("\n" + "=" * 60)
