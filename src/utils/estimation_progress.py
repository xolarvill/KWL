"""
专门为估计工作流设计的轻量级进度管理器
使用 pickle 序列化来支持复杂数据类型（如numpy数组）
"""
import pickle
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class EstimationProgressTracker:
    """专门为估计工作流优化的轻量级进度跟踪器"""
    
    def __init__(self, 
                 progress_dir: str = "progress",
                 task_name: str = "estimation",
                 save_interval: int = 300):  # 默认5分钟
        """
        初始化进度跟踪器
        
        Args:
            progress_dir: 进度文件保存目录
            task_name: 任务名称
            save_interval: 保存间隔时间（秒），默认5分钟
        """
        self.progress_dir = Path(progress_dir)
        self.task_name = task_name
        self.save_interval = save_interval
        
        # 加入时间戳的进度文件命名规则（使用pickle格式）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = self.progress_dir / f"{task_name}_progress_{timestamp}.pkl"
        
        # 添加EM算法特定的进度保存机制
        self.last_em_save_time = time.time()
        self.em_save_interval = save_interval  # EM算法保存间隔
        
        # 确保进度目录存在
        self.progress_dir.mkdir(exist_ok=True)
        
        # 状态
        self.state = {
            'task_name': task_name,
            'current_phase': '',
            'completed_phases': [],
            'phase_results': {},
            'start_time': None,
            'last_update': None,
            'step_count': 0,
            'is_resumed': False
        }
        
    def load_state(self) -> bool:
        """加载状态 - 查找最新的进度文件"""
        # 如果没有找到当前时间戳的文件，尝试查找最新的进度文件
        if not self.progress_file.exists():
            # 查找相同任务名称的所有进度文件
            pattern = f"{self.task_name}_progress_*.pkl"
            existing_files = list(self.progress_dir.glob(pattern))
            
            if existing_files:
                # 按修改时间排序，取最新的文件
                latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                self.progress_file = latest_file
                logger.info(f"找到最新的进度文件: {latest_file.name}")
            else:
                return False
            
        try:
            with open(self.progress_file, 'rb') as f:
                loaded_state = pickle.load(f)
                
            # 验证必要字段
            required_fields = ['task_name', 'completed_phases', 'phase_results']
            if not all(field in loaded_state for field in required_fields):
                logger.warning("进度文件格式无效")
                return False
                
            self.state.update(loaded_state)
            self.state['is_resumed'] = True
            
            logger.info(f"加载进度: 已完成 {len(self.state['completed_phases'])} 个阶段")
            logger.info(f"已完成阶段: {self.state['completed_phases']}")
            return True
            
        except Exception as e:
            logger.warning(f"加载进度失败: {e}")
            return False
    
    def save_state(self, force: bool = False):
        """保存状态"""
        current_time = time.time()
        
        # 检查是否应该保存（基于时间间隔）
        if not force and self.state['last_update'] and \
           (current_time - self.state['last_update']) < self.save_interval:
            return
            
        try:
            self.state['last_update'] = current_time
            if self.state['start_time'] is None:
                self.state['start_time'] = current_time
                
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.debug(f"进度已保存: {self.state['current_phase']}")
            
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def start_phase(self, phase_name: str, description: str = ""):
        """开始新阶段"""
        self.state['current_phase'] = phase_name
        self.state['step_count'] += 1
        
        if self.state['start_time'] is None:
            self.state['start_time'] = time.time()
            
        logger.info(f"[{self.state['step_count']}] 开始阶段: {phase_name}")
        if description:
            logger.info(f"  └─ {description}")
            
        # 定期保存
        if self.state['step_count'] % self.save_interval == 0:
            self.save_state()
    
    def complete_phase(self, phase_name: str, result: Any = None):
        """完成阶段"""
        if phase_name not in self.state['completed_phases']:
            self.state['completed_phases'].append(phase_name)
            
        if result is not None:
            # pickle 可以序列化大多数Python对象，包括numpy数组
            try:
                self.state['phase_results'][phase_name] = result
            except Exception as e:
                logger.warning(f"阶段 '{phase_name}' 的结果无法序列化，跳过保存: {e}")
                
        logger.info(f"✓ 完成阶段: {phase_name}")
        self.save_state()
    
    def is_phase_completed(self, phase_name: str) -> bool:
        """检查阶段是否已完成"""
        return phase_name in self.state['completed_phases']
    
    def get_phase_result(self, phase_name: str) -> Any:
        """获取阶段结果"""
        return self.state['phase_results'].get(phase_name)
    
    def should_save_em_progress(self, force: bool = False) -> bool:
        """检查是否应该保存EM算法进度"""
        current_time = time.time()
        if force or (current_time - self.last_em_save_time) >= self.em_save_interval:
            self.last_em_save_time = current_time
            return True
        return False
    
    def save_em_progress(self, em_state: Dict[str, Any], force: bool = False):
        """保存EM算法中间状态"""
        if self.should_save_em_progress(force):
            try:
                # 保存EM算法特定的状态
                if 'em_progress' not in self.state:
                    self.state['em_progress'] = {}
                
                self.state['em_progress'].update({
                    'current_params': em_state.get('current_params'),
                    'current_pi_k': em_state.get('current_pi_k'),
                    'current_log_likelihood': em_state.get('current_log_likelihood'),
                    'iteration': em_state.get('iteration'),
                    'converged': em_state.get('converged', False),
                    'last_update': time.time()
                })
                
                self.save_state(force=True)  # 强制保存EM进度
                logger.info(f"EM进度已保存: 第{em_state.get('iteration', 0)}次迭代, "
                           f"对数似然: {em_state.get('current_log_likelihood', 0):.6f}")
                
            except Exception as e:
                logger.error(f"保存EM进度失败: {e}")
    
    def get_em_progress(self) -> Optional[Dict[str, Any]]:
        """获取保存的EM算法进度"""
        return self.state.get('em_progress')
    
    def get_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        elapsed = time.time() - (self.state['start_time'] or time.time())
        return {
            'task_name': self.state['task_name'],
            'total_phases': len(self.state['completed_phases']),
            'current_phase': self.state['current_phase'],
            'elapsed_time': elapsed,
            'is_resumed': self.state['is_resumed'],
            'completed_phases': self.state['completed_phases'],
            'has_em_progress': 'em_progress' in self.state,
            'last_em_iteration': self.state.get('em_progress', {}).get('iteration') if 'em_progress' in self.state else None
        }
    
    def cleanup(self, cleanup_all: bool = False):
        """清理进度文件
        
        Args:
            cleanup_all: 是否清理所有同任务的进度文件，还是只清理当前文件
        """
        try:
            if cleanup_all:
                # 清理所有同任务的进度文件
                pattern = f"{self.task_name}_progress_*.pkl"
                existing_files = list(self.progress_dir.glob(pattern))
                for file_path in existing_files:
                    file_path.unlink()
                    logger.info(f"清理进度文件: {file_path.name}")
            else:
                # 只清理当前文件
                if self.progress_file.exists():
                    self.progress_file.unlink()
                    logger.info(f"进度文件已清理: {self.progress_file.name}")
        except Exception as e:
            logger.warning(f"清理进度文件失败: {e}")


@contextmanager
def estimation_progress(task_name: str = "estimation", 
                       progress_dir: str = "progress",
                       save_interval: int = 5,
                       auto_cleanup: bool = False):
    """
    估计工作流进度跟踪的上下文管理器
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        save_interval: 保存间隔（秒）
        auto_cleanup: 完成后是否自动清理进度文件
    
    Yields:
        EstimationProgressTracker: 进度跟踪器实例
    """
    tracker = EstimationProgressTracker(
        progress_dir=progress_dir,
        task_name=task_name,
        save_interval=save_interval
    )
    
    # 加载已有进度
    has_progress = tracker.load_state()
    
    try:
        yield tracker
        
    except Exception as e:
        # 异常时强制保存
        logger.error(f"估计任务异常退出: {e}")
        tracker.save_state(force=True)
        raise
        
    finally:
        if auto_cleanup:
            tracker.cleanup(cleanup_all=False)  # 只清理当前任务的当前文件


def resume_estimation_phase(tracker: EstimationProgressTracker, 
                          phase_name: str,
                          phase_func: callable,
                          *args, **kwargs):
    """
    恢复或执行估计阶段
    
    Args:
        tracker: 进度跟踪器
        phase_name: 阶段名称
        phase_func: 阶段执行函数
        *args, **kwargs: 传递给阶段函数的参数
    
    Returns:
        阶段执行结果
    """
    # 检查是否已完成
    if tracker.is_phase_completed(phase_name):
        logger.info(f"阶段 '{phase_name}' 已完成，跳过执行")
        saved_result = tracker.get_phase_result(phase_name)
        if saved_result is not None:
            return saved_result
        else:
            logger.warning(f"阶段 '{phase_name}' 已完成但没有保存结果，重新执行")
    
    # 执行阶段
    tracker.start_phase(phase_name)
    
    try:
        result = phase_func(*args, **kwargs)
        tracker.complete_phase(phase_name, result)
        return result
        
    except Exception as e:
        logger.error(f"阶段 '{phase_name}' 执行失败: {e}")
        tracker.save_state(force=True)
        raise


def get_estimation_progress(task_name: str = "estimation", 
                          progress_dir: str = "progress") -> Optional[Dict[str, Any]]:
    """获取估计任务进度摘要"""
    tracker = EstimationProgressTracker(
        progress_dir=progress_dir, 
        task_name=task_name
    )
    
    if tracker.load_state():
        return tracker.get_summary()
    return None


def get_latest_progress_file(task_name: str = "estimation", 
                           progress_dir: str = "progress") -> Optional[Path]:
    """获取最新的进度文件路径
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        
    Returns:
        最新的进度文件路径，如果不存在则返回None
    """
    progress_path = Path(progress_dir)
    pattern = f"{task_name}_progress_*.pkl"
    existing_files = list(progress_path.glob(pattern))
    
    if not existing_files:
        return None
    
    # 按修改时间排序，取最新的文件
    latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
    return latest_file


def load_latest_progress(task_name: str = "estimation", 
                        progress_dir: str = "progress") -> Optional[Dict[str, Any]]:
    """加载最新的进度数据
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        
    Returns:
        最新的进度数据，如果不存在则返回None
    """
    latest_file = get_latest_progress_file(task_name, progress_dir)
    if latest_file is None:
        return None
    
    try:
        with open(latest_file, 'rb') as f:
            progress_data = pickle.load(f)
        logger.info(f"成功加载最新进度文件: {latest_file.name}")
        return progress_data
    except Exception as e:
        logger.error(f"加载进度文件失败 {latest_file.name}: {e}")
        return None


def resume_from_latest_progress(task_name: str = "estimation", 
                               progress_dir: str = "progress") -> Optional[Dict[str, Any]]:
    """从最新的进度恢复EM算法状态
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        
    Returns:
        包含EM算法状态的字典，如果无法恢复则返回None
    """
    progress_data = load_latest_progress(task_name, progress_dir)
    if progress_data is None:
        return None
    
    em_progress = progress_data.get('em_progress')
    if em_progress is None:
        logger.warning("进度文件中没有EM算法状态数据")
        return None
    
    # 检查是否有完整的EM状态
    required_fields = ['current_params', 'current_pi_k', 'current_log_likelihood', 'iteration']
    if not all(field in em_progress for field in required_fields):
        logger.warning("EM进度数据不完整")
        return None
    
    logger.info(f"可以从进度恢复: 第{em_progress['iteration']}次迭代, "
               f"对数似然: {em_progress['current_log_likelihood']:.6f}")
    return em_progress


def cleanup_old_progress_files(task_name: str = "estimation", 
                               progress_dir: str = "progress", 
                               keep_latest: int = 5):
    """清理旧的进度文件，保留最新的几个
    
    Args:
        task_name: 任务名称
        progress_dir: 进度文件目录
        keep_latest: 保留最新的文件数量
    """
    progress_path = Path(progress_dir)
    pattern = f"{task_name}_progress_*.pkl"
    existing_files = list(progress_path.glob(pattern))
    
    if len(existing_files) <= keep_latest:
        logger.info(f"进度文件数量 ({len(existing_files)}) 不超过保留数量 ({keep_latest})，无需清理")
        return
    
    # 按修改时间排序
    existing_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 删除旧的文件
    files_to_delete = existing_files[keep_latest:]
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            logger.info(f"删除旧进度文件: {file_path.name}")
        except Exception as e:
            logger.warning(f"删除文件失败 {file_path.name}: {e}")
    
    logger.info(f"已清理 {len(files_to_delete)} 个旧进度文件，保留 {keep_latest} 个最新文件")
