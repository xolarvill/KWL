#!/usr/bin/env python3
"""
进度管理工具脚本
用于查看、清理和管理估计任务的进度
"""
import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.estimation_progress import get_estimation_progress
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_progress():
    """检查当前进度状态"""
    # 检查默认任务名称的进度
    progress = get_estimation_progress(task_name="main_estimation")
    
    if progress:
        print("\n=== 当前进度状态 ===")
        print(f"任务名称: {progress['task_name']}")
        print(f"已完成阶段数: {progress['total_phases']}")
        print(f"当前阶段: {progress['current_phase'] or '无'}")
        print(f"运行时间: {progress['elapsed_time']:.1f} 秒")
        print(f"恢复模式: {'是' if progress['is_resumed'] else '否'}")
        
        if progress['completed_phases']:
            print(f"\n已完成的阶段:")
            for i, phase in enumerate(progress['completed_phases'], 1):
                print(f"  {i}. {phase}")
    else:
        print("未找到进度文件")
        
        # 也检查是否有其他任务的进度文件
        progress_dir = Path("progress")
        if progress_dir.exists():
            progress_files = list(progress_dir.glob("*_progress.json"))
            if progress_files:
                print(f"\n找到 {len(progress_files)} 个其他任务的进度文件:")
                for pf in progress_files:
                    print(f"  - {pf.stem.replace('_progress', '')}")


def clean_progress():
    """清理进度文件"""
    import shutil
    
    progress_dir = Path("progress")
    if not progress_dir.exists():
        print("进度目录不存在")
        return
    
    # 查找所有进度文件
    progress_files = list(progress_dir.glob("*_progress.json"))
    
    if not progress_files:
        print("未找到进度文件")
        return
    
    print(f"\n找到 {len(progress_files)} 个进度文件:")
    for pf in progress_files:
        print(f"  - {pf.name}")
    
    response = input("\n确定要删除这些文件吗? (y/N): ").strip().lower()
    
    if response == 'y':
        try:
            for pf in progress_files:
                pf.unlink()
                print(f"已删除: {pf.name}")
            print("进度文件清理完成")
        except Exception as e:
            print(f"清理失败: {e}")
    else:
        print("取消清理操作")


def list_all_progress():
    """列出所有进度文件"""
    progress_dir = Path("progress")
    if not progress_dir.exists():
        print("进度目录不存在")
        return
    
    progress_files = list(progress_dir.glob("*_progress.json"))
    
    if not progress_files:
        print("未找到进度文件")
        return
    
    print(f"\n找到 {len(progress_files)} 个进度文件:")
    for pf in progress_files:
        # 尝试获取简要信息
        try:
            import json
            with open(pf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task_name = data.get('task_name', 'unknown')
            completed = len(data.get('completed_phases', []))
            current = data.get('current_phase', 'none')
            
            print(f"\n  📋 {pf.name}")
            print(f"     任务: {task_name}")
            print(f"     已完成阶段: {completed}")
            print(f"     当前阶段: {current}")
            
        except Exception as e:
            print(f"\n  📋 {pf.name} (无法读取: {e})")


def main():
    parser = argparse.ArgumentParser(description="进度管理工具")
    
    parser.add_argument('command', 
                       choices=['check', 'clean', 'list'],
                       help='要执行的命令: check(检查进度), clean(清理进度), list(列出所有进度文件)')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_progress()
    elif args.command == 'clean':
        clean_progress()
    elif args.command == 'list':
        list_all_progress()


if __name__ == '__main__':
    main()