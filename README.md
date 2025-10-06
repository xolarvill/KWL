# Estimation Log Visualizer

一个用于分析和可视化估计程序日志的工具，可以帮您更轻松地理解估计过程、定位问题和分析性能。

## 功能特性

- **Web 界面**: 提供直观的网页界面用于日志分析
- **命令行界面**: 快速命令行分析功能
- **过滤功能**: 按日志级别、迭代次数、搜索文本等过滤日志
- **摘要报告**: 生成估计过程的统计摘要
- **性能分析**: 提供性能分析和时间线视图
- **实时搜索**: 快速定位关键信息

## 安装依赖

```bash
cd /path/to/KWL
uv sync
```

## 使用方法

### Web 界面（推荐）

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

# 导出过滤后的日志
uv run python -m src.main /path/to/logfile.log --search "ERROR" --export filtered_output.log
```

## Web 界面功能说明

1. **日志文件选择**：从 `progress/log/` 目录中选择要分析的日志文件
2. **过滤选项**：
   - 设置最低日志级别
   - 搜索特定文本
   - 选择显示/隐藏缓存操作
   - 选择显示/隐藏 Bellman 操作
3. **标签页**：
   - **摘要**：显示估计过程的整体统计信息
   - **时间线**：按时间顺序显示日志条目
   - **性能**：分析估计过程的性能指标
   - **过滤日志**：显示根据过滤条件筛选的日志

## 命令行参数

- `--web`: 启动 Web 界面
- `--port PORT`: 指定 Web 界面端口（默认 8080）
- `--summary`: 生成摘要报告
- `--performance`: 生成性能分析
- `--timeline`: 生成时间线视图
- `--filter-level LEVEL`: 设置最低日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- `--search TEXT`: 搜索包含特定文本的日志
- `--max-entries N`: 设置时间线视图最大显示条数
- `--export FILE`: 将过滤后的日志导出到文件

## 分析指标

- **迭代次数**: EM 算法完成的迭代次数
- **收敛情况**: 算法是否收敛
- **缓存效率**: 缓存命中率
- **Bellman 方程**: 收敛迭代次数
- **对数似然**: 每次迭代的对数似然值
- **类型概率**: 每种类型的概率分布
- **执行时间**: 各阶段的耗时分析

## 文件结构

```
src/
├── log_visualization/   # 日志可视化模块
│   ├── __init__.py
│   ├── log_parser.py    # 日志解析器
│   ├── log_visualizer.py # 日志可视化器
│   ├── web_visualizer.py # Web 界面
│   └── cli_visualizer.py # 命令行界面
└── main.py             # 主应用入口
```