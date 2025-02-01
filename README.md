# README.md

## Graduation Thesis Code Snippets

This project implements an economic model simulation using Python. The model is designed to analyze and simulate economic behaviors based on various parameters and data inputs.

## Project Structure

```bash
│  .gitnore
│  main.py # 主运行文件
│  README.md # 说明文件
│  requirements.txt # 需求
├─data
│  │  cfps10_22mc.dta # 个人数据
│  │  linguistic.json # 汉藏语系谱系树
│  └─geo
│      │  2000-2022年296个地级以上城市房价数据.xlsx # 房价数据
│      │  geo.xlsx # 地理信息数据
│      │  adjacent.xlsx # 现成的邻近矩阵
└─function
    │  adjacent.py # 邻近矩阵
    │  calK.py # 最近访问地区序列
    │  compare_vec.py # 比较向量是否相同
    │  data_geo.py # 读取地理信息文件
    │  data_person.py # 读取个人信息文件
    │  descriptive.py # 描述性统计
    │  distance.py # 距离矩阵
    │  houseprice.py # 房价
    │  indicator.py # Boolean指示器
    │  linguistic.py # 语言LCA距离
    │  llh_individual.py # 传统方法的个人llh建模
    │  llh_individual_ds.py # class封装的个人llh建模
    │  llh_log_sample.py # 传统方法的样本llh建模
    │  llh_log_sample_ds.py # class封装的样本llh建模
    │  method_entropy.py # 熵值法
    │  method_pca.py # PCA主成分分析法
    │  nelder_mead.py # Nelder-mead检验方法
    │  nelder_mead1.py # 另一种Nelder-mead方法
    │  newton_line_search.py # LU优化的牛顿法
    │  optimal.py # 拟牛顿法优化
    │  std.py # 求标准误
    │  subsample.py # 样本分割
    │  tradition.py # 传统方法求取代估参数
```

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
