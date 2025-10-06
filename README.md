# 基于非参混合动态离散选择模型分析中国2010-2020年居民移动以及使用ABM模型进行反事实政策模拟

本项目使用 Python 实现了一个非参数混合的动态离散选择模型，用于分析中国的劳动力迁移决策。该模型受到 Kennan 和 Walker（2011）框架的启发，在其模型基础上引入了对户籍制度效应，并且使推力和拉力都在模型中得到了充分体现。

**请注意：项目中，build是使用LaTeX编译论文的单独项目，不与其他文件产生联动。**

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


## 程序成果清单

### 模块一：DDCM 核心估计模块 (对应 `\chapter{估计结果}`)

这是所有分析的基础，目标是得到一组稳健的、可解释的微观行为参数。

**1. 核心结构参数表 (`\ref{tab:结构参数的估计结果}`):**
-   **内容**:
    -   **效用函数参数**: `\alpha_w` (收入边际效用), `\alpha_s` (各项舒适度amenities的系数), `\alpha_{home}` (家乡溢价), `\lambda` (损失厌恶系数, 如果加入)。
    -   **户籍惩罚函数参数**: `\theta_{base}`, `\theta_{edu}`, `\theta_{health}`, `\theta_{house}`。
    -   **迁移成本函数参数**: `\gamma_1` (距离), `\gamma_2` (邻接), `\gamma_3` (历史), `\gamma_4` (年龄), `\gamma_5` (规模)。
-   **格式**: 参数名 | 估计值 (Estimate) | 标准误 (Std. Error) | 显著性标记 (*, **, ***)。

**2. 有限混合模型结果表 (`\ref{tab:有限混合参数的 ઉ估计结果}`):**
-   **内容**:
    -   **类型占比**: `\pi_1`, `\pi_2`, `\pi_3` ...
    -   **类型特定的迁移成本**: `\gamma_{0, \tau=1}`, `\gamma_{0, \tau=2}`, `\gamma_{0, \tau=3}` ...
-   **格式**: 类型 (Type) | 占比 (Share, `\pi_\tau`) | 固定迁移成本 (Fixed Cost, `\gamma_{0\tau}`) | (对应的标准误)。
-   **作用**: 揭示人口中存在哪些不同的“行为模式”群体，并量化其规模。例如，您可能会发现一个占比60%的“高成本定居型”和一个占比10%的“低成本闯荡型”。

**3. 未观测异质性分布图/表:**
-   **内容**:
    *   对于 `\eta_i`, `\nu_{ij}`, `\xi_{ij}`, `\sigma_{\varepsilon,i}`，报告您估计出的**支撑点位置**。
-   **格式**:
    -   **表格**: 异质性来源 | 支撑点1 | 支撑点2 | ...
    -   **图形 (更佳)**: 画出这些支撑点的**直方图或核密度图**，直观展示这些未观测属性的分布形状（是否偏态、双峰等）。
-   **作用**: 展示您的非参数方法捕捉到的复杂分布形态，证明了放弃简单正态假设的必要性。

**4. 模型拟合检验结果:**
-   **内容**:
    -   **汇总指标**: 对数似然值, AIC/BIC, Brier Score。
    -   **关键矩的拟合图/表**:
        -   **图1 (生命周期迁移率)**: X轴为年龄，Y轴为迁移概率。一条线是真实数据，另一条线是模型模拟结果。
        -   **图2 (迁移流向矩阵/弦图)**: 展示模型模拟的主要省际人口流动与真实数据的对比。
        -   **表 (返乡率/省内迁移占比)**: 比较模型模拟值与真实数据值。
    -   **样本外预测精度表**: 报告用前期数据估计的模型在后期数据上的预测准确率 (Hit Rate) 或其他指标。
-   **作用**: 用证据告诉审稿人：“我的模型不仅理论复杂，而且能很好地拟合真实世界的数据。”

**5. 机制分解结果图 (`\sec:机制分解}`):**
-   **内容**:
    *   通常是一个**条形图 (Bar Chart)**。
    -   **Y轴**: 某个宏观结果，例如“全国平均迁移率”或“地区间人均收入差距”。
    -   **X轴**: 不同的反事实情景。
        *   Bar 1: 基准模型 (Baseline Model)。
        *   Bar 2: 无户籍惩罚 (No Hukou Penalty)。
        *   Bar 3: 无家乡溢价 (No Home Premium)。
        *   Bar 4: 无地理成本 (No Distance Cost)。
-   **作用**: 这是您DDCM部分故事性的高潮，直观地量化了不同摩擦对宏观格局的贡献。


### 模块二：ABM 宏观模拟模块 (对应 `\chapter{与宏观桥接}`)

这是将您的研究从小世界推向大世界的关键，目标是展示模型的宏观解释力和政策模拟能力。

**6. ABM模型校准结果表:**
-   **内容**:
    -   **目标矩**: 您选择用于校准的10-30个宏观矩（如省际净流入率、返乡率等）。
    -   **真实值 vs 模拟值**: 清晰地列出每个矩的真实数据值，以及您校准好的ABM模型模拟出的均值和95%置信区间。
-   **格式**: 目标矩 (Target Moment) | 真实数据 (Data) | 模型模拟均值 (Model Mean) | 模型模拟95% CI。
-   **作用**: 证明您的ABM平台是经过“宏观事实”锚定的，其模拟结果具有外部有效性。

**7. 宏观涌现模式验证图 (`\sub:abm模型的检验}`):**
-   **内容**:
    -   **图1 (城市规模分布)**: Log-log图，X轴为城市排名的对数，Y轴为城市人口的对数。一条线是真实数据，另一条是ABM模拟结果，看是否都近似一条直线（齐普夫定律）。
    -   **图2 (人口空间分布图)**: 两张中国地图，一张用真实数据染色显示各省人口密度，另一张用ABM模拟的长期均衡结果染色。看是否能再现“胡焕庸线”等宏观格局。
-   **作用**: 展示您的模型不仅能拟合您用来校准的矩，还能“涌现”出一些您并未直接校准的、更高层次的宏观规律。这是模型强大的标志。

**8. 核心政策实验结果图/表 (`\section{反事实参数估计}`):**
-   **这是您ABM部分故事性的高潮，也是政策建议的核心依据。**
-   **内容**:
    *   对于每个政策实验（如“二线城市吸引力提升”），输出一系列**动态演化图 (Time-path Plots)**。
    -   **图1 (主要城市人口路径)**: X轴为年份（从2022到2042），Y轴为人口。图中包含几条线：北京（基准 vs 政策）、上海（基准 vs 政策）、成都（基准 vs 政策）、武汉（基准 vs 政策）。
    -   **图2 (全国不平等指数路径)**: X轴为年份，Y轴为基尼/泰尔指数。一条线是基准情景，另一条是政策情景。
    -   **图3 (社会福利路径)**: X轴为年份，Y轴为加总的个体效用或等价收入。
-   **作用**: 将政策影响从一个静态的数字变成一个动态的故事，展示短期、中期、长期的不同效果以及可能的非预期后果。

**9. 稳健性与权衡分析图:**
-   **内容**:
    -   **图 (Pareto前沿)**: 如您所设想的，X轴为“地区均衡”（如人口集中度下降），Y轴为“总福利/产出”。图中的每个点代表一种政策强度或组合，连接起来形成效率与公平的权衡曲线。
    -   **表/图 (敏感性分析)**: 展示当外部性参数 `η_1, η_2` 等变化时，政策实验的核心结论是否依然成立。
-   **作用**: 展示您对模型和政策复杂性的深刻理解，表明您的政策建议不是“一招鲜”，而是存在复杂的权衡取舍。

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
