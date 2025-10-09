# ModelConfig 参数管理系统使用指南

## 一、ModelConfig 的作用

`ModelConfig` 类是整个项目的**统一参数管理中心**，负责：

1. ✅ 所有数据文件路径
2. ✅ 所有算法超参数（EM算法、Bootstrap等）
3. ✅ 所有结构参数的初始值
4. ✅ 类型特定参数（type-specific parameters）
5. ✅ 提供便捷的参数访问接口

## 二、当前参数状态总览

### 1. 数据路径 ✅
所有数据文件路径已在`ModelConfig`中定义

### 2. 算法超参数 ✅

| 参数组 | 参数名 | 默认值 | 说明 |
|--------|--------|--------|------|
| EM算法 | `em_max_iterations` | 100 | 最大迭代次数 |
| | `em_tolerance` | 1e-4 | 收敛容差 |
| | `em_n_types` | 3 | 混合模型类型数 |
| L-BFGS-B | `lbfgsb_maxiter` | 15 | M-step优化最大迭代 |
| | `lbfgsb_gtol` | 1e-3 | 梯度容差 |
| | `lbfgsb_ftol` | 1e-3 | 函数值容差 |
| Bootstrap | `bootstrap_n_replications` | 200 | Bootstrap重复次数 |
| | `bootstrap_max_em_iter` | 5 | 每次Bootstrap的EM迭代 |
| | `bootstrap_seed` | 42 | 随机种子 |
| | `bootstrap_n_jobs` | -1 | 并行核心数 |

### 3. 结构参数初始值 ✅

#### 共享参数（所有类型共用）
- `alpha_w = 1.0` - 收入效用
- `rho_base_tier_1 = 1.0` - 户籍惩罚
- `rho_edu = 0.1` - 户籍×教育
- `rho_health = 0.1` - 户籍×医疗
- `rho_house = 0.1` - 户籍×住房
- `alpha_climate = 0.1` - 气候舒适度
- `alpha_education = 0.1` - 教育舒适度
- `alpha_health = 0.1` - 医疗舒适度
- `alpha_public_services = 0.1` - 公共服务舒适度
- `gamma_1 = -0.1` - 距离对迁移成本的影响
- `gamma_2 = 0.2` - 邻近性影响
- `gamma_3 = -0.4` - 回流迁移影响
- `gamma_4 = 0.01` - 年龄影响
- `gamma_5 = -0.05` - 人口规模影响

#### 类型特定参数

| 参数 | Type 0 (机会型) | Type 1 (稳定型) | Type 2 (适应型) |
|------|----------------|----------------|----------------|
| `gamma_0` (固定迁移成本) | 0.1 | 5.0 | 1.5 |
| `gamma_1` (距离敏感性) | -0.5 | -3.0 | -1.5 |
| `alpha_home` (家乡溢价) | 0.1 | 2.0 | 0.8 |
| `lambda` (损失厌恶) | 2.5 | 1.2 | 1.8 |

## 三、如何使用ModelConfig

### 方法1：获取初始参数字典

```python
from src.config.model_config import ModelConfig

config = ModelConfig()

# 获取所有初始参数（包括type-specific）
params = config.get_initial_params(use_type_specific=True)
# 返回：{'alpha_w': 1.0, 'gamma_0_type_0': 0.1, ...}

# 获取EM算法配置
em_config = config.get_em_config()
# 返回：{'max_iterations': 100, 'tolerance': 1e-4, ...}

# 获取Bootstrap配置
bootstrap_config = config.get_bootstrap_config()
# 返回：{'n_bootstrap': 200, 'max_em_iterations': 5, ...}
```

### 方法2：修改参数值

```python
# 修改单个参数
config.update_param('gamma_0_type_0', 0.2)

# 批量修改
config.alpha_w = 1.5
config.bootstrap_n_replications = 500
```

### 方法3：在EM算法中使用

```python
from src.estimation.em_nfxp import run_em_algorithm
from src.config.model_config import ModelConfig

config = ModelConfig()
em_conf = config.get_em_config()

results = run_em_algorithm(
    observed_data=data,
    state_space=state_space,
    transition_matrices=transition_matrices,
    regions_df=regions_df,
    distance_matrix=distance_matrix,
    adjacency_matrix=adjacency_matrix,
    **em_conf  # 展开EM配置
)
```

## 四、已更新的模块

### ✅ 已完成

1. **`src/config/model_config.py`**
   - 完全重写
   - 添加了所有参数
   - 提供了便捷方法

2. **`src/estimation/migration_behavior_analysis.py`**
   - `create_behavior_based_initial_params()` 接受可选的 `config` 参数
   - 如果提供config，直接使用；否则使用fallback默认值

3. **`src/estimation/em_nfxp.py`**
   - 支持通过 `initial_params` 和 `initial_pi_k` 参数传入自定义初始值
   - Bootstrap可以正确使用原始估计结果作为初始值

4. **`src/estimation/inference.py`**
   - Bootstrap函数已修复
   - 数值Hessian已修复（排除n_choices）

### ⚠️ 需要手动更新的文件

1. **`scripts/02_run_estimation.py`**
   ```python
   # 建议修改：
   from src.config.model_config import ModelConfig

   config = ModelConfig()

   estimation_params = {
       "observed_data": df_individual,
       "regions_df": df_region,
       "state_space": state_space,
       "transition_matrices": transition_matrices,
       "distance_matrix": distance_matrix,
       "adjacency_matrix": adjacency_matrix,
       **config.get_em_config()  # 使用ModelConfig的EM配置
   }
   ```

2. **`scripts/03_test_inference.py`**
   ```python
   # 使用ModelConfig提供的Bootstrap配置
   from src.config.model_config import ModelConfig

   config = ModelConfig()
   bootstrap_config = config.get_bootstrap_config()

   bootstrap_standard_errors(
       ...,
       **bootstrap_config
   )
   ```

## 五、参数命名规范

### 1. 文件路径
- 小写，用下划线分隔
- 示例：`individual_data_path`, `distance_matrix_path`

### 2. 算法超参数
- 前缀标识算法：`em_`, `bootstrap_`, `lbfgsb_`
- 示例：`em_max_iterations`, `bootstrap_n_replications`

### 3. 结构参数
- 希腊字母用英文：`alpha`, `beta`, `gamma`, `lambda`, `rho`
- 下标用下划线：`alpha_w`, `gamma_0`
- Type-specific加后缀：`gamma_0_type_0`, `alpha_home_type_1`

### 4. 类型特定参数
- 格式：`{param_name}_type_{type_id}`
- 示例：`gamma_0_type_0`, `lambda_type_2`

## 六、快速检查清单

在运行Bootstrap测试前，确保：

- [x] ModelConfig定义了所有需要的参数
- [x] migration_behavior_analysis.py支持config参数
- [x] em_nfxp.py支持initial_params参数
- [x] Bootstrap函数传递initial_params
- [x] 数值Hessian排除了n_choices
- [ ] 02_run_estimation.py使用ModelConfig ← **你可以手动更新**
- [ ] 03_test_inference.py使用ModelConfig ← **你可以手动更新**

## 七、测试命令

```bash
# 1. 测试ModelConfig
uv run python src/config/model_config.py

# 2. 小样本Bootstrap测试（推荐）
uv run python scripts/03_test_inference.py --test 2 --sample-size 50 --n-bootstrap 10

# 3. 正式Bootstrap（耗时较长）
uv run python scripts/03_test_inference.py --test 2 --sample-size 500 --n-bootstrap 200
```

## 八、注意事项

1. **参数初始值的选择**
   - 当前值是基于文献和经验设定的
   - 可以根据你的数据特征调整
   - Type-specific参数应该有明显差异（便于识别类型）

2. **Bootstrap参数**
   - `n_bootstrap=200` 是标准选择
   - 如果时间允许，可以增加到500
   - `max_em_iterations=5` 对于Bootstrap足够（因为从估计值附近开始）

3. **并行计算**
   - `n_jobs=-1` 使用所有CPU核心
   - 如果系统不稳定，可以设为具体数字（如4或8）

---

**你现在可以开始Bootstrap测试了！** 🚀
