"""
实验管理器
用于管理和跟踪训练实验
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
    def create_experiment(self, 
                         name: str, 
                         config: Dict,
                         description: str = "",
                         tags: List[str] = None) -> str:
        """创建新实验"""
        
        # 创建实验目录
        exp_dir = self.experiments_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        
        # 保存配置
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 保存实验元数据
        metadata = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "config": config
        }
        
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 实验 '{name}' 创建成功")
        print(f"📁 实验目录: {exp_dir}")
        
        return str(exp_dir)
    
    def list_experiments(self) -> List[Dict]:
        """列出所有实验"""
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
                else:
                    # 兼容旧格式
                    experiments.append({
                        "name": exp_dir.name,
                        "description": "旧实验格式",
                        "created_at": "未知",
                        "status": "未知"
                    })
        
        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_experiment(self, name: str) -> Optional[Dict]:
        """获取实验信息"""
        exp_dir = self.experiments_dir / name
        metadata_file = exp_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def update_experiment_status(self, name: str, status: str):
        """更新实验状态"""
        exp_dir = self.experiments_dir / name
        metadata_file = exp_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            metadata["status"] = status
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def delete_experiment(self, name: str, confirm: bool = False):
        """删除实验"""
        exp_dir = self.experiments_dir / name
        
        if not exp_dir.exists():
            print(f"❌ 实验 '{name}' 不存在")
            return
        
        if not confirm:
            print(f"⚠️  确定要删除实验 '{name}' 吗？")
            response = input("输入 'yes' 确认: ")
            if response.lower() != 'yes':
                print("取消删除")
                return
        
        shutil.rmtree(exp_dir)
        print(f"✅ 实验 '{name}' 已删除")
    
    def archive_experiment(self, name: str):
        """归档实验"""
        exp_dir = self.experiments_dir / name
        archive_dir = self.experiments_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        if exp_dir.exists():
            shutil.move(str(exp_dir), str(archive_dir / name))
            print(f"📦 实验 '{name}' 已归档")
        else:
            print(f"❌ 实验 '{name}' 不存在")
    
    def print_experiments_summary(self):
        """打印实验摘要"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("📝 暂无实验")
            return
        
        print(f"📊 实验总数: {len(experiments)}")
        print("=" * 80)
        
        for i, exp in enumerate(experiments, 1):
            print(f"{i}. {exp['name']}")
            print(f"   描述: {exp.get('description', '无')}")
            print(f"   状态: {exp.get('status', '未知')}")
            print(f"   创建时间: {exp.get('created_at', '未知')}")
            
            # 显示标签
            tags = exp.get('tags', [])
            if tags:
                print(f"   标签: {', '.join(tags)}")
            
            print()

def main():
    """命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("action", choices=["create", "list", "delete", "archive", "summary"],
                       help="操作类型")
    parser.add_argument("--name", type=str, help="实验名称")
    parser.add_argument("--description", type=str, default="", help="实验描述")
    parser.add_argument("--tags", type=str, nargs="+", help="实验标签")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--confirm", action="store_true", help="确认删除")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.action == "create":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        
        config = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        
        manager.create_experiment(
            args.name, 
            config, 
            args.description,
            args.tags
        )
    
    elif args.action == "list":
        experiments = manager.list_experiments()
        for exp in experiments:
            print(f"- {exp['name']} ({exp.get('status', '未知')})")
    
    elif args.action == "delete":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.delete_experiment(args.name, args.confirm)
    
    elif args.action == "archive":
        if not args.name:
            print("❌ 请提供实验名称")
            return
        manager.archive_experiment(args.name)
    
    elif args.action == "summary":
        manager.print_experiments_summary()

if __name__ == "__main__":
    main()
