# 基于非参混合动态离散选择模型分析中国2010-2020年居民移动以及使用ABM模型进行反事实政策模拟

本项目使用 Python 实现了一个非参数混合的动态离散选择模型，用于分析中国的劳动力迁移决策。该模型受到 Kennan 和 Walker（2011）框架的启发，在其模型基础上引入了对户籍制度效应，并且使推力和拉力都在模型中得到了充分体现。

**请注意：项目中，build是使用LaTeX编译论文的单独子项目，不与其他文件产生联动。**

## 使用说明

使用`git clone`或者下载文件压缩包并解压后在终端中输入
```bash
cd (相应的路径)
uv sync
source .venv/bin/activate
uv run main.py
```

也可以单独运行`scripts/`文件夹下的运行文件


## 项目结构

### 数据构成与形式

- `data/processed/geo.xlsx` 是地区相关数据，格式为平衡面板数据，示例如下：

| provcd | prov_name | year | 常住人口（万)	| 人均可支配收入（元） | ... |
|--------|--------|------|--------|-----------|-----|
| 11     | 北京市    | 2010 | 4.2       | 20 |--------|
| 11     | 北京市   | 2011  | 6.2       | 30 |--------|
| 21     | 辽宁省 | 2010   | 4.7       | 30 |--------|
| 21     | 辽宁省 | 2011    | 4.9       | 40 |--------|
| ...    | ...  | ...    | ...       | ... |--------|


- `data/processed/clds.csv` 是个体追踪数据，格式为非平衡面板数据，示例如下：

| IID | year | provcd | age | income | hukou_prov | is_at_hukou|... |
|-----|------|--------|-----|------|-----|----|----|
| 1   | 2010 | 110000     | 20  | 1000 | 110000 |1|----|
| 1   | 2011 | 110000 | 21  | 1200 | 310000 |0|...|
| 2   | 2010 | 110000 | 20  | 2100 | 110000|0|...|
| 2   | 2011 | 110000 | 21  | 2600 | 350000 |1|...|
| ... | ...  | ...    | ... | ...  | ... |...|...|

此外，`data/processed` 目录还包括一些矩阵文件，例如 `adjacent_matrix.xlsx` 为邻接矩阵.

此外，我将附上一些数据中变量对应到论文中具体内容的表格
|文件|变量名|论文中的变量|论文中的意义|
|---|---|---|---|
|clds.csv|IID|i|个体标识|
|clds.csv|year|t|时间|
|clds.csv|provcd|j|所在地区|
|clds.csv|hukou_prov|hukou|个体i的户口地区|
|clds.csv|hometown|home|个体i的家乡|
|clds.csv|is_at_hukou|I(j=hukou_i)|个体i当前是否在户口地|
|clds.csv|income|w_itj|个体i在t期j地时的收入|
|clds.csv|age|a_it|个体i在t期的年龄|
|distance_matrix.xlsx|本身是矩阵|Dis|表示两个地区之间的距离|
|adjacent_matrix.xlsx|本身是矩阵|Adj|表示两个地区是否相邻|
|linguistic_distance_matrix.csv|本身是矩阵|Similarity|用语言距离表示两个省份之间的文化亲近度|
|geo_amenities.csv|amenity_climater|Climate|气候，属于amenities的一部分|
|geo_amenities.csv|amenity_education|Edu|教育，属于amenities的一部分|
|geo_amenities.csv|自然灾害受灾人口（万人）|Hazzard|自然灾害，属于amenities的一部分|
|geo_amenities.csv|amenity_|Education|教育，属于amenities的一部分|
|geo_amenities.csv|房价收入比|HousingPrice|房价收入比，属于amenities的一部分|
|geo_amenities.csv|amenity_public_services|Public|公共服务，属于amenities的一部分|
|geo_amenities.csv|amenity_health|Health|医疗，属于amenities的一部分|
|geo.xlsx|移动电话普及率|Internet|互联网接入度|
|geo.xlsx|常住人口万|n_jt|j地区在t期的人口|
|geo.xlsx|地区基本经济面|mu_jt|地区的相对基本工资|
|geo.xlsx|户籍难度等级|Tier|获取当地户口的难度|

*注：geo_amenities.csv需要通过`scripts/00_prepare_data.py`生成，clds_preprocessed_with_wages.csv需要通过`scripts/01_train_ml_plugins.py`生成。*

## 估计日志

每条日志代表一个agent_type的价值函数求解

agent_type不是"个体"，而是"类型"
- agent_type=0  # Type 0的所有个体共享的价值函数
- agent_type=1  # Type 1的所有个体共享的价值函数
- agent_type=2  # Type 2的所有个体共享的价值函数

Bellman方程求解的是状态空间级别的，每次求解产生一个价值函数向量 $V^k(s)$：
$$V^k(s) = \max_{j \in \mathcal{J}} \left \{ u^k(s,j) + \beta \sum_{s'} P(s'|s,j) V^k(s') \right \}$$

其中：
- $k$ = agent_type（类型0, 1, 2）
- $s$ = 状态（年龄、前期位置等的组合）
- 你的状态空间有 1,488个状态

所以一次Bellman求解产生：converged_v.shape = (1488)

### 具体例子说明
#### E-step 过程
```python
# E-step: 为15,864个个体计算类型后验概率
for individual_i in 15864_individuals:
    for k in [0, 1, 2]:  # 三种类型
        # 计算该个体在类型k下的似然
        # 使用类型k的价值函数V^k（已缓存，所有个体共享）
        likelihood_i_k = compute_likelihood(individual_i, V_k)
```
  关键点：
  - 15,864个个体 × 3个类型 = 47,592次似然计算
  - 但只需要求解 3个Bellman方程（每个类型1个）
  - 这就是为什么你只看到3条"Cache MISS"

#### M-step 过程（L-BFGS-B优化）

当L-BFGS-B评估目标函数时：
```python
def objective_function(params):
    # L-BFGS-B会用不同的参数多次调用这个函数
    for k in [0, 1, 2]:
        # 提取类型k的参数
        params_k = extract_type_k_params(params)

        # 求解Bellman方程（或使用缓存）
        V_k = solve_bellman(params_k, agent_type=k)  # ← Cache MISS/HIT在这里

        # 用V_k计算所有个体的加权似然
        for individual in all_individuals:
            likelihood += weight[individual, k] * log_p(individual | V_k)
```

### 查看实际的日志模式:

第1次评估（初始参数）:
    ├─ Type 0 [MISS] ← 首次求解，49次迭代
    ├─ Type 1 [MISS] ← 首次求解，49次迭代
    └─ Type 2 [MISS] ← 首次求解，49次迭代

第2次评估（参数略微调整）:
    ├─ Type 0 [MISS] ← 参数变了，缓存失效
    ├─ Type 1 [MISS]
    └─ Type 2 [MISS]

第3次评估:
    ├─ Type 0 [MISS]
    ├─ Type 1 [MISS]
    └─ Type 2 [MISS]

第4次评估:
    ├─ Type 0 [MISS]
    ├─ Type 1 [MISS]
    └─ Type 2 [MISS]

第5次评估（L-BFGS-B计算梯度，回到某个已评估的参数）:
    ├─ Type 0 [HIT] ← 缓存命中！
    ├─ Type 1 [HIT]
    └─ Type 2 [HIT]

#### 为什么有这么多MISS？

L-BFGS-B是拟牛顿法，它会：
1. 尝试不同的参数组合（探索）
2. 计算梯度（需要重新评估某些点）
3. 进行线搜索（沿着搜索方向尝试不同步长）


| 概念        | 数量       | 说明             |
|-----------|----------|----------------|
| 个体        | 15,864   | 实际观测的农民工       |
| 观测        | 32,826   | 个体×时期的决策记录     |
| 类型        | 3        | 潜在的异质性类别       |
| 状态        | 1,488    | (年龄, 前期位置) 的组合 |
| Bellman求解 | 每次3个     | 每个类型1个价值函数     |
| 似然计算      | 32,826×3 | 每个观测×每个类型      |

每条"Cache MISS for agent_type=X"日志表示：
- 为类型X求解一个包含1,488个状态的价值函数
- 这个价值函数会被用于计算该类型下所有15,864个个体的似然


## LOG可视化文件阅读

### Web 界面

```bash
# 启动 Web 界面
uv run python -m src.main --web --port 8080

# 然后在浏览器中访问 http://127.0.0.1:8080
```

Web 界面提供以下功能：
- 选择和加载日志文件
- 摘要报告
- 时间线视图
- 性能分析
- 过滤和搜索功能

### 命令行界面

```bash
# 显示日志摘要
uv run python -m src.main /path/to/logfile.log --summary

# 显示性能分析
uv run python -m src.main /path/to/logfile.log --performance

# 显示时间线视图（最多显示50条）
uv run python -m src.main /path/to/logfile.log --timeline --max-entries 100

# 搜索包含特定文本的日志
uv run python -m src.main /path/to/logfile.log --search "Cache HIT" --filter-level INFO

# 对比两个日志文件
uv run python -m src.main /path/to/logfile1.log /path/to/logfile2.log --compare

# 导出过滤后的日志
uv run python -m src.main /path/to/logfile.log --search "ERROR" --export filtered_output.log
```

### 命令行参数

- `--web`: 启动 Web 界面
- `--port PORT`: 指定 Web 界面端口（默认 8080）
- `--summary`: 生成摘要报告
- `--performance`: 生成性能分析
- `--timeline`: 生成时间线视图
- `--filter-level LEVEL`: 设置最低日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `--search TEXT`: 搜索包含特定文本的日志
- `--max-entries N`: 设置时间线视图最大显示条数
- `--compare`: 对比两个日志文件
- `--export FILE`: 将过滤后的日志导出到文件

### 分析指标

- **迭代次数**: EM 算法完成的迭代次数
- **收敛情况**: 算法是否收敛
- **缓存效率**: 缓存命中率
- **Bellman 方程**: 收敛迭代次数
- **对数似然**: 每次迭代的对数似然值
- **类型概率**: 每种类型的概率分布
- **执行时间**: 各阶段的耗时分析

### 对比功能

- **基本统计对比**: 总条目数、错误数、警告数等
- **性能对比**: 执行时间、缓存效率等
- **收敛对比**: 两个日志的收敛情况
- **关键指标差异**: 对数似然、迭代次数等的差异
