# KWL-Thesis: A Dynamic Discrete Choice Model of Labor Migration

This project implements a dynamic discrete choice model in Python to analyze labor migration decisions in China, with a particular focus on the impact of the Hukou (户籍) system. The model is inspired by the framework of Kennan and Walker (2011) and is the core of my graduation thesis.

This README provides a guide to understanding the project's structure, core economic model, and how to run the estimation.

## Core Economic Model & Key Assumptions

The model analyzes an individual's decision to migrate by comparing the expected lifetime utility of staying in the current province versus moving to a new one.

### Utility Function

An individual's utility in a given province `j` is composed of several parts:
1.  **Economic Utility**: Primarily driven by expected wages, which are influenced by age, regional economic conditions, and individual-specific unobserved skills (`eta`, `nu`).
2.  **Amenity Utility**: The non-monetary value of living in a province, including factors like housing prices, environment, education, and healthcare.
3.  **Migration Costs**: The costs associated with moving, which depend on distance, provincial borders, age, and other factors.
4.  **Hukou (户籍) Penalty**: This is a key component of the model. We hypothesize that individuals face a utility penalty if they choose to reside in a province where they do not hold a Hukou. This is captured by the **`alphaP`** parameter. A negative and statistically significant `alphaP` would provide evidence for the restrictive effect of the Hukou system on labor mobility.

The model is solved using backward induction via dynamic programming to find the optimal migration strategy for each individual over their lifecycle.

## Project Structure

The project is organized into several key directories:

```
.
├── main.py               # Main script to run the entire estimation process
├── README.md             # This file
├── config/               # Configuration files
│   └── model_config.py   # Manages all model parameters and initial values
├── data/                 # Scripts for data loading and preprocessing
│   ├── data_loader.py    # Main data loader class
│   ├── data_person.py    # Cleans and prepares individual panel data (CFPS)
│   └── data_region.py    # Cleans and prepares regional characteristics data
├── estimation/           # Parameter estimation and statistical inference
│   └── model_estimator.py# The core estimator class using PyTorch and L-BFGS
├── file/                 # Raw data files (e.g., .dta, .xlsx, .json)
├── model/                # Core model implementation
│   ├── dynamic_model.py  # Defines the overall dynamic model structure
│   ├── individual_likelihood.py # Calculates the likelihood for a single individual
│   └── migration_parameters.py  # Manages all model parameters as torch.nn.Parameter
├── outputs_logs/         # Directory for output files and logs
└── utils/                # Utility functions (e.g., indicators, data processing helpers)
```

## 数据构成与形式

`file/geo.xlsx`为地区相关数据，格式为面板数据，其内容格式示例如下：

|provcd|year|health|education|...|
|-|-|-|-|-|
|11|2010|4.1|4.2|...|
|11|2011|6.1|6.2|...|
|21|2010|4.5|4.7|...|
|21|2011|5.4|4.9|...|
|...|...|...|...|...|

其中`provcd`为省份代码，`year`为年份，`health`为卫生指数，`education`为教育指数。

`file/cfps10_20.dta`为个体数据，格式为面板数据，其内容格式示例如下：

|pid|year|provcd|age|wage|...|
|-|-|-|-|-|-|
|1|2010|11|20|1000|...|
|1|2011|11|21|1200|...|
|2|2010|11|20|2100|...|
|2|2011|11|21|2600|...|
|...|...|...|...|...|...|

其中`pid`为个体标识，`year`为年份，`provcd`为省份代码，`age`为年龄，`sex`为性别。

file中还包括了一些矩阵文件，例如`file/adjacent.xlsx`为邻接矩阵文件；又有一些数据文件用json格式储存，如`file/linguistic.json`为语言谱系树文件。

## Implementation Methods

**Overall Objective**: Implement dynamic programming using PyTorch vectorization and optimize structural parameters using automatic differentiation.

1. **Parameter Management**: The `ModelConfig` class in `config/model_config.py` manages the initial parameter guesses, while the `MigrationParameters` class in `model/migration_parameters.py` manages the parameters as variables of type `torch.nn.Parameter`.
2. **Parameters to be Estimated**: The parameters include not only structural parameters but also some variables related to discretized support points.
3. **Algorithm Framework**: The algorithm still employs **Backward Induction**, with optimizations achieved through vectorized and matrix-based methods.
4. **Standard Error Calculation**: The `ModelEstimator` in `estimation/model_estimator` computes standard errors using PyTorch's automatic differentiation capabilities.
5. **Hypothesis Testing**: The `ModelEstimator` performs Wald tests based on calculated standard errors to obtain p-values.
6. **Goodness-of-Fit Evaluation**: The `ModelEstimator` computes log-likelihood values using both AIC and Vuong Test methods, and also calculates McFadden's Pseudo-R² as a goodness-of-fit metric.
7. **Prediction Accuracy and Simulation Calibration**: The `ModelEstimator` includes functionality for prediction accuracy and simulation calibration. Prediction accuracy compares predicted outcomes with actual observations, while simulation calibration generates simulated data using estimated model parameters. Since this functionality is not yet finalized, `ModelEstimator.prediction_accuracy()` and `ModelEstimator.simulated_calibration()` are designed to support a degree of customization.

## Specific Code Implementation Logic

- The `ModelConfig` class in `config/model_config.py` is responsible for managing all parameters.
- The `DataLoader` class in `data/data_loader.py` handles returning the loaded data.
- The `IndividualLikelihood` class in `model/individual_likelihood.py` accepts individual-specific data and region-related data to compute the individual likelihood function.
- The `DynamicModel` class in `model/dynamic_model.py` splits the panel data by `pid`, passes each individual’s data into `IndividualLikelihood` to compute their individual likelihoods (with computation efficiency improved via `joblib`), and then aggregates these individual likelihoods into the sample likelihood function.
- The `ModelEstimator` class in `estimation/model_estimator` receives the sample likelihood function and performs parameter estimation, returning the estimated parameter values.
- The `ModelInference` class in `estimation/model_inference.py` implements standard error calculation, p-value computation, goodness-of-fit evaluation, prediction accuracy, and simulation calibration, and returns the final results.
- The full workflow is orchestrated in `main.py`.

The implementation flow should follow this chain:

> `DataLoader` → `DynamicModel` receives data and returns the sample likelihood function (by internally calling `IndividualLikelihood` to compute individual likelihood functions, which in turn use `DynamicProgramming`) → `ModelEstimator` receives the sample likelihood function and performs parameter estimation, standard error calculation, p-value computation, goodness-of-fit evaluation, prediction accuracy, and simulation calibration → Return the final results.

## Code Logic Flow

The estimation process follows a clear chain of command:

1.  **`main.py`**: The entry point that orchestrates the entire process.
2.  **`config.ModelConfig`**: Loads all initial configurations and parameters.
3.  **`data.DataLoader`**: Loads and preprocesses individual and regional data from the `file/` directory.
4.  **`model.DynamicModel`**:
    *   Receives the data from the `DataLoader`.
    *   Initializes the `DynamicProgramming` solver to calculate the value function `EV`.
    *   Splits the data by individual (`pid`) and uses `joblib` to parallelize the calculation of individual likelihoods.
    *   For each individual, it instantiates `model.IndividualLikelihood`.
5.  **`model.IndividualLikelihood`**:
    *   Takes a single individual's data.
    *   Crucially, it uses the individual's **`hukou`** information when calculating choice probabilities.
    *   It calls the `DynamicProgramming` solver, passing the `hukou` data to correctly calculate the utility, including any penalties.
    *   It returns the likelihood contribution for that single individual.
6.  **`estimation.ModelEstimator`**:
    *   Receives the aggregated log-likelihood function from `DynamicModel`.
    *   Uses the `L-BFGS` optimizer via `torch.optim` to find the parameters that maximize the log-likelihood.
    *   Leverages `torch.autograd` to automatically compute gradients.
    *   After optimization, it calculates standard errors (via the Hessian matrix), p-values, and other fit statistics (AIC, BIC).
7.  **`main.py`**: Calls the estimator to save the final results and generate a report in LaTeX format.

## How to Run

1.  **Install Dependencies**: This project is based on `uv` for environment management.
   ```bash
   uv venv
   uv sync
   ```

2.  **Configure Data Paths**: Before running, make sure the file paths in `config/model_config.py` point to the correct locations of your data files in the `file/` directory.

3.  **Run Estimation**: Execute the main script from the project root directory.
    ```bash
    python main.py
    ```
    The script will print progress to the console and save detailed logs and output tables in the `outputs_logs/` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

# 分析中国劳动力迁移的动态离散选择模型

本项目使用 Python 实现了一个动态离散选择模型，用于分析中国的劳动力迁移决策，特别关注**户籍 (Hukou)**制度的影响。该模型受到 Kennan 和 Walker（2011）框架的启发，是本人毕业论文的核心内容。

本 README 提供了对项目结构、核心经济模型以及如何运行估计过程的指南。

## 核心经济模型与关键假设

模型通过比较个体在当前省份停留或迁移到新省份的预期终身效用，来分析其迁移决策。

### 效用函数

个体在某个省份 `j` 的效用由以下几个部分组成：

- **经济效用**：主要由工资决定，而工资受年龄、地区经济状况以及个体未观测到的能力 (`eta`, `nu`) 影响。
- **宜居效用**：居住地区的非货币价值，包括住房价格、环境、教育和医疗等因素。
- **迁移成本**：与距离、跨省边界、年龄等相关的迁移代价。
- **户籍 (Hukou) 惩罚**：这是模型的关键部分。我们假设，如果个体居住在其没有户籍的省份，则会面临效用惩罚。这一惩罚由参数 `alphaP` 表示。若 `alphaP` 为负且统计显著，则表明户籍制度对劳动力流动具有限制作用。

模型通过**反向归纳法**（Backward Induction）结合动态规划求解，以找到个体生命周期内的最优迁移策略。

## 项目结构

项目主要分为以下目录：

```
.
├── main.py               # 主程序，用于运行整个估计流程
├── README.md             # 本文件
├── config/               # 配置文件
│   └── model_config.py   # 管理所有模型参数和初始值
├── data/                 # 数据加载与预处理脚本
│   ├── data_loader.py    # 主数据加载类
│   ├── data_person.py    # 清洗并准备个体面板数据（CFPS）
│   └── data_region.py    # 清洗并准备地区特征数据
├── estimation/           # 参数估计与统计推断
│   └── model_estimator.py# 使用 PyTorch 和 L-BFGS 的核心估计器类
├── file/                 # 原始数据文件（如 .dta, .xlsx, .json）
├── model/                # 核心模型实现
│   ├── dynamic_model.py  # 定义整体动态模型结构
│   ├── individual_likelihood.py # 计算单个个体的似然函数
│   └── migration_parameters.py  # 将所有模型参数管理为 torch.nn.Parameter
├── outputs_logs/         # 输出文件与日志目录
└── utils/                # 工具函数（如指标计算、数据处理辅助）
```

## 数据构成与形式

- `file/geo.xlsx` 是地区相关数据，格式为面板数据，示例如下：

| provcd | year | health | education | ... |
|--------|------|--------|-----------|-----|
| 11     | 2010 | 4.1    | 4.2       | ... |
| 11     | 2011 | 6.1    | 6.2       | ... |
| 21     | 2010 | 4.5    | 4.7       | ... |
| 21     | 2011 | 5.4    | 4.9       | ... |
| ...    | ...  | ...    | ...       | ... |

其中：
- `provcd`：省份代码
- `year`：年份
- `health`：卫生指数
- `education`：教育指数

- `file/cfps10_20.dta` 是个体数据，格式为面板数据，示例如下：

| pid | year | provcd | age | wage | ... |
|-----|------|--------|-----|------|-----|
| 1   | 2010 | 11     | 20  | 1000 | ... |
| 1   | 2011 | 11     | 21  | 1200 | ... |
| 2   | 2010 | 11     | 20  | 2100 | ... |
| 2   | 2011 | 11     | 21  | 2600 | ... |
| ... | ...  | ...    | ... | ...  | ... |

其中：
- `pid`：个体标识符
- `year`：年份
- `provcd`：省份代码
- `age`：年龄
- `wage`：工资

此外，`file` 目录还包括一些矩阵文件，例如 `file/adjacent.xlsx` 为邻接矩阵；还有一些 JSON 文件，如 `file/linguistic.json` 为语言谱系树文件。


## 实现方法

**总体目标**：使用 PyTorch 向量化实现动态规划，并利用自动微分优化结构参数。

**参数管理方式**：
- 使用 `config/model_config.py` 中的 `ModelConfig` 类管理初始猜测值。
- 使用 `model/migration_parameters.py` 中的 `MigrationParameters` 类管理参数，这些参数为 `torch.nn.Parameter` 类型变量。

**算法框架**：仍然采用**反向归纳法**（Backward Induction），但通过向量和矩阵化方法进行优化。

**标准误计算**：
- `estimation/model_estimator` 中的 `ModelEstimator` 利用 PyTorch 的自动微分功能计算标准误。

**假设检验**：
- `ModelEstimator` 使用 Wald 检验，基于标准误计算 p 值。

**拟合优度评估**：
- 使用 AIC 和 Vuong Test 两种方法计算对数似然值。
- 同时计算 McFadden's Pseudo-R² 作为拟合度指标。

**预测准确率与模拟校准**：
- `ModelEstimator.prediction_accuracy()` 和 `ModelEstimator.simulated_calibration()` 支持一定程度的自定义化功能，目前仍在完善中。

## 特定代码实现逻辑

- `config/model_config.py` 中的 `ModelConfig` 负责管理所有参数。
- `data/data_loader.py` 中的 `DataLoader` 负责返回数据内容。
- `model/individual_likelihood.py` 中的 `IndividualLikelihood` 接收单个个体数据和其他地区数据，从而计算该个体的似然函数。
- `model/dynamic_model.py` 中的 `DynamicModel` 负责按 `pid` 分割个体面板数据，并传入 `IndividualLikelihood` 获取个体似然函数，最终整合成样本似然函数，使用 `joblib` 提高计算效率。
- `estimation/model_estimator` 中的 `ModelEstimator` 接收样本似然函数，进行参数估计并返回结果。
- `estimation/model_inference.py` 中的 `ModelInference` 实现标准误、p 值、拟合优度、预测准确率和模拟校准的计算，并输出最终结果。

**整个代码流程链如下**：

> DataLoader -> DynamicModel 接收数据并返回样本似然函数（通过 IndividualLikelihood 返回个体似然函数实现，其中包含 DynamicProgramming）-> ModelEstimator 接收样本似然函数，进行参数估计、标准误计算、p 值计算、拟合优度计算、预测准确率、模拟校准 -> 返回最终结果

## 代码执行流程

- `main.py`：入口点，负责协调整个流程。
- `config.ModelConfig`：加载所有初始配置和参数。
- `data.DataLoader`：从 `file/` 目录加载并预处理个体和区域数据。
- `model.DynamicModel`：
  - 接收来自 `DataLoader` 的数据。
  - 初始化 `DynamicProgramming` 求解器，计算期望效用 `EV`。
  - 按照 `pid` 分割数据，使用 `joblib` 并行计算个体似然。
  - 对每个个体实例化 `model.IndividualLikelihood`。
- `model.IndividualLikelihood`：
  - 接收单个个体的数据。
  - 在计算选择概率时使用个体的 `hukou` 信息。
  - 调用 `DynamicProgramming` 求解器，传递 `hukou` 数据以正确计算效用（包括惩罚项）。
  - 返回该个体的似然贡献。
- `estimation.ModelEstimator`：
  - 接收来自 `DynamicModel` 的汇总对数似然函数。
  - 使用 `L-BFGS` 优化器通过 `torch.optim` 寻找最大化对数似然的参数。
  - 利用 `torch.autograd` 自动计算梯度。
  - 优化完成后，计算标准误（通过 Hessian 矩阵）、p 值及其他拟合统计量（AIC、BIC）。
- `main.py`：调用估计器保存最终结果并生成 LaTeX 报告。

## 如何运行

1. **安装依赖**：本项目使用 `uv` 进行环境管理。
   ```bash
   uv venv
   uv sync
   ```

2. **配置数据路径**：运行前请确保 `config/model_config.py` 中的数据路径指向 `file/` 目录下的正确文件位置。

3. **运行估计**：从项目根目录执行主脚本：
   ```bash
   python main.py
   ```
   脚本将在控制台打印进度，并将详细日志和输出表格保存至 `outputs_logs/` 目录。

## License

该项目采用 MIT License，请参阅 [LICENSE](LICENSE) 文件了解详细信息。