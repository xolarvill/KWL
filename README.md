# README.md

## Graduation Thesis Code Snippets

This project implements an economic model simulation using Python. The model is designed to analyze and simulate economic behaviors based on various parameters and data inputs. The weather data collection is performed by using my other repository: [Hisotry Weather Data Collection](github.com/xolarvill/history_weather_data_collection).

## Project Structure

```bash
│  .gitignore
│  main.py # 主运行文件
│  README.md # 说明文件
│  requirements.txt # 需求
├─build # latex编译pdf
├─file # 数据文件
├─outputs_logs # 输出和日志文件
├─config
    │  
├─data
    │  adjacent.py # 邻近矩阵
    │  data_geo.py # 读取地理信息文件
    │  data_person.py # 读取个人信息文件
    │  distance.py # 距离矩阵
    │  houseprice.py # 房价
    │  linguistic.py # 语言LCA距离
    │  method_entropy.py # 熵值法
    │  method_pca.py # PCA主成分分析法
├─models
    │  calK.py # 最近访问地区序列
    │  llh_individual.py # 传统方法的个人llh建模
    │  llh_individual_ds.py # class封装的个人llh建模
    │  llh_log_sample.py # 传统方法的样本llh建模
    │  llh_log_sample_ds.py # class封装的样本llh建模
├─optimization
    │  subsample.py # 样本分割
    │  nelder_mead.py # Nelder-mead检验方法
    │  nelder_mead1.py # 另一种Nelder-mead方法
    │  newton_line_search.py # LU优化的牛顿法
    │  optimal.py # 拟牛顿法优化
    │  tradition.py # 传统方法求取代估参数
├─utils
    │  indicator.py # Boolean指示器
    │  compare_vec.py # 比较向量是否相同
    │  descriptive.py # 描述性统计
    │  std.py # 求标准误
│   .gitignore
│  main.py # 主运行文件
│  README.md # 说明文件
│  requirements.txt # 需求
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

## 实现方法
总体目标：使用PyTorch向量化实现动态规划，利用自动微分优化结构参数

1. 参数管理方法：使用`config/model_config.py`中的`ModelConfig`类管理初始猜测值，使用`model/migration_parameters.py`中的`MigrationParameters`类管理参数，参数具体为`torch.nn.Parameter`类型的变量。
2. 代估计参数不仅包含结构参数，还包括离散化支撑点相关的一些变量。
3. 算法框架仍是反向归纳(Backward Induction)，使用向量\矩阵化方法优化算法
4. 标准误计算：`estimation/model_estimator`中`ModelEstimator`利用PyTorch的自动微分特性计算标准误
5. 假设检验：`estimation/model_estimator`中`ModelEstimator`使用Wald检验，基于标准误计算p值
6. 拟合优度评估：`estimation/model_estimator`中`ModelEstimator`用AIC和Vuong Test两种方法计算对数似然值，并且计算McFadden's Pseudo-R²作为拟合度指标。
7. `estimation/model_estimator`中`ModelEstimator`包含预测准确率和模拟校准两项功能。预测准确率指将预测结果与实际观测值进行比较，模拟校准指使用估计的模型参数，生成模拟数据。由于该功能暂时未完全定稿，ModelEstimator.prediction_accuracy()和ModelEstimator.simlulated_calibration()需要能支持一定程度的自定义化。

## 特定代码实现逻辑
- `config/model_config.py`中的`ModelConfig`负责管理一切参数
- `data/data_loader.py`中的`DataLoader`负责返回数据内容
- `model/individual_likelihood.py`中的`IndividualLikelihood`负责接受单个个体数据和其他地区相关的数据，从而得到该个体的个体似然函数
- `model/dynamic_model.py`中的`DynamicModel`需要负责将个体面板数据按照`pid`分割，将分割好的个体数据传入至`IndividualLikelihood`中从而获取个体的似然函数使用`joblib`优化计算效率；并且需要将个体似然函数整合成样本似然函数
- `estimation/model_estimator`中的`ModelEstimator`需要接受样本似然函数，进行参数估计，返回参数估计值
- `estimation/model_inference.py`中的`ModelInference`实现标准误计算、p值计算、拟合优度计算、预测准确率、模拟校准，并返回最终结果
- 最终在`main.py`中实现全流程

代码实现的链条应该如下：
DataLoader -> DynamicModel接受数据，返回样本似然函数 (通过内部的IndividualLikelihood返回个体似然函数实现，而Individual Likelihood中则包含DynamicProgramming) -> ModelEstimator接受样本似然函数，进行参数估计、标准误计算、p值计算、拟合优度计算、预测准确率、模拟校准 -> 返回最终结果


## Installation

To run this project, ensure you have Python installed along with the required dependencies. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

To execute the simulation, run the following command:

```bash
python main.py
```

This will initialize the parameters, read the necessary data, and execute the main functions of the economic model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
