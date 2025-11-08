import os
import json
import heapq
import random
from typing import List, Tuple, Optional, Dict, Any


class Config:
    """配置类，可存储任意JSON可序列化的配置数据"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        return cls(**data)


class CodeManager:
    def __init__(self, data_dir: str):
        """
        初始化管理器
        :param data_dir: 数据存储目录路径
        """
        self.data_dir = data_dir
        self.data = {}  # {id: (code, score, id, config)}
        self.next_id = 1
        self.index_file = os.path.join(data_dir, "index.json")

        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 加载现有数据
        self._load_data()

    def _load_data(self):
        """从目录加载所有数据"""
        # 加载索引文件（如果存在）
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                self.next_id = index_data.get('next_id', 1)
                records = index_data.get('records', [])

                for record in records:
                    id = record['id']
                    file_path = os.path.join(self.data_dir, f"{id}.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as data_file:
                            data = json.load(data_file)
                            config = Config.from_dict(data['config'])
                            self.data[id] = (
                                data['code'],
                                data['score'],
                                id,
                                config
                            )

    def _save_index(self):
        """保存索引文件"""
        records = [{'id': id} for id in self.data.keys()]
        index_data = {
            'next_id': self.next_id,
            'records': records
        }
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _save_record(self, id: int):
        """保存单个记录到文件"""
        if id in self.data:
            code, score, _, config = self.data[id]
            record = {
                'code': code,
                'score': score,
                'config': config.to_dict()
            }
            file_path = os.path.join(self.data_dir, f"{id}.json")
            with open(file_path, 'w') as f:
                json.dump(record, f, indent=2)

    def add(self, code: str, score: float = 1.0) -> int:
        """
        添加新条目
        :param code: 代码文本
        :param score: 初始分数
        :return: 分配的新ID
        """
        new_id = self.next_id
        self.next_id += 1
        self.data[new_id] = (code, score, new_id, Config())
        self._save_record(new_id)
        self._save_index()
        return new_id

    def get(self, id: int) -> Optional[Tuple[str, float, int, Config]]:
        """
        获取指定ID的条目
        :param id: 条目ID
        :return: (code, score, id, config) 或 None
        """
        return self.data.get(id, None)

    def update(self, id: int, config: Optional[Config] = None, score: Optional[float] = None):
        """
        更新条目
        :param id: 要更新的条目ID
        :param config: 新的配置（可选）
        :param score: 新的分数（可选）
        """
        if id not in self.data:
            raise ValueError(f"ID {id} not found")

        code, old_score, _, old_config = self.data[id]
        new_config = config if config is not None else old_config
        new_score = score if score is not None else old_score

        self.data[id] = (code, new_score, id, new_config)
        self._save_record(id)

    def top_k(self, k: int) -> List[Tuple[str, float, int, Config]]:
        """
        使用轮盘赌选择算法选择k个条目
        :param k: 要返回的条目数量
        :return: 选中的条目列表
        """
        # 获取所有模型
        models = list(self.data.values())
        if k <= 0:
            return []

        # 如果模型数量不足k，则返回所有
        if len(models) <= k:
            return models

        # 计算总分数
        total_score = sum(score for _, score, _, _ in models)

        # 如果所有分数都是0，则随机选择
        if total_score == 0:
            return random.sample(models, k)

        # 计算选择概率
        probabilities = [score / total_score for _, score, _, _ in models]

        # 确保概率总和为1（处理浮点精度问题）
        probabilities[-1] = 1 - sum(probabilities[:-1])

        # 使用轮盘赌选择k个模型
        selected_indices = random.choices(
            range(len(models)),
            weights=probabilities,
            k=k
        )

        # 获取选中的模型（可能有重复，但概率很低）
        selected_models = [models[i] for i in selected_indices]

        return selected_models

    def save_all(self):
        """显式保存所有数据（通常自动保存，但提供额外控制）"""
        for id in self.data:
            self._save_record(id)
        self._save_index()