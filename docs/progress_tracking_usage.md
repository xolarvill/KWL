# 进度跟踪和断点续跑功能使用说明

## 功能概述

为 `scripts/02_run_estimation.py` 添加了轻量级的进度跟踪和断点续跑功能，具有以下特点：

- **自动检测**：启动时自动检测是否存在未完成的进度
- **断点续跑**：支持从上次中断的地方继续执行
- **轻量级设计**：使用上下文管理器，不影响主逻辑
- **定期保存**：每10秒自动保存一次进度，避免频繁I/O
- **异常保护**：异常退出时确保保存当前状态
- **灵活配置**：可通过命令行参数控制功能开关

## 主要组件

### 1. 进度管理类 (`src/utils/estimation_progress.py`)

- `EstimationProgressTracker`: 核心进度跟踪器
- `estimation_progress()`: 上下文管理器
- `resume_estimation_phase()`: 阶段恢复/执行函数

### 2. 进度管理脚本 (`scripts/manage_progress.py`)

用于查看和管理进度状态的独立脚本

### 3. 主脚本集成 (`scripts/02_run_estimation.py`)

- 新增进度跟踪参数
- 支持断点续跑的工作流
- 向后兼容（可禁用进度跟踪）

## 使用方法

### 基本使用

```bash
# 正常运行（启用进度跟踪）
python scripts/02_run_estimation.py

# 禁用进度跟踪（传统模式）
python scripts/02_run_estimation.py --no-progress-tracking

# 运行完成后自动清理进度文件
python scripts/02_run_estimation.py --auto-cleanup-progress
```

### 进度管理

```bash
# 检查当前进度状态
python scripts/manage_progress.py check

# 列出所有进度文件
python scripts/manage_progress.py list

# 清理所有进度文件
python scripts/manage_progress.py clean
```

### 主脚本进度检查

```bash
# 检查进度状态并退出
python scripts/02_run_estimation.py --check-progress

# 清理所有进度文件
python scripts/02_run_estimation.py --clean-progress
```

## 进度文件结构

进度文件保存在 `progress/` 目录下，文件名格式为 `{task_name}_progress.json`：

```json
{
  "task_name": "main_estimation",
  "current_phase": "model_estimation", 
  "completed_phases": ["data_preparation"],
  "phase_results": {...},
  "start_time": 1730472054.123,
  "last_update": 1730472074.456,
  "step_count": 2,
  "is_resumed": true
}
```

## 工作流阶段

当前估计工作流分为以下阶段：

1. **data_preparation**: 数据加载和准备
2. **model_estimation**: 模型估计（EM算法）
3. **standard_error_computation**: 标准误计算（Louis/Bootstrap/数值方法）
4. **result_output**: 结果输出和保存

## 断点续跑示例

### 场景1：正常中断后恢复

```bash
# 第一次运行，在模型估计阶段中断
python scripts/02_run_estimation.py
# ... 运行中断 ...

# 第二次运行，自动从模型估计阶段开始
python scripts/02_run_estimation.py
# 输出: "阶段 'data_preparation' 已完成，跳过执行"
```

### 场景2：异常中断后恢复

```bash
# 运行过程中发生异常
python scripts/02_run_estimation.py
# ... 异常退出 ...

# 重新运行，自动恢复到最后保存的状态
python scripts/02_run_estimation.py
```

## 注意事项

1. **结果保存**：只有可JSON序列化的结果才会被保存，复杂对象会跳过保存
2. **内存使用**：进度文件只保存关键信息，不保存大型数据对象
3. **兼容性**：进度文件格式可能会随版本更新，建议定期清理旧进度
4. **并行执行**：当前不支持多进程并行执行的进度跟踪

## 故障排除

### 进度文件损坏

如果进度文件损坏，可以手动删除：

```bash
rm progress/main_estimation_progress.json
```

或使用清理命令：

```bash
python scripts/manage_progress.py clean
```

### 进度不更新

检查是否有写权限，进度文件是否被其他进程锁定。

### 恢复失败

如果恢复失败，可能是由于：
- 代码版本不兼容
- 数据结构发生变化
- 依赖环境发生变化

解决方法是清理进度文件后重新运行。

## 性能影响

- **时间开销**：每次保存进度约需10-50ms
- **存储开销**：每个进度文件约1-10KB
- **内存开销**：进度跟踪器内存占用<1MB

总体性能影响可以忽略不计。

## 扩展开发

### 添加新阶段

在 `_run_estimation_with_tracking()` 函数中添加：

```python
new_result = resume_estimation_phase(
    tracker, "new_phase_name", new_phase_function, *args
)
```

### 自定义保存间隔

```python
with estimation_progress(
    task_name="custom_task",
    save_interval=30  # 每30秒保存一次
) as tracker:
    # ... 工作代码 ...
```

### 自定义进度目录

```python
with estimation_progress(
    task_name="custom_task",
    progress_dir="custom_progress_dir"
) as tracker:
    # ... 工作代码 ...
```