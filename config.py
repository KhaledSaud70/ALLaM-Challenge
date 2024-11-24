from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from pathlib import Path
import yaml

@dataclass
class LLMConfig:
    provider: Literal["custom", "openai", "anthropic"]
    name: str
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.provider == "custom" and self.name not in ["allam-13b", "FakeChatModel"]:
            raise ValueError(f"Invalid custom model name: {self.name}. Must be 'allam-13b' or 'FakeChatModel'")

@dataclass
class ModelPaths:
    meter_weights_path: Path
    diacritizer_weights_path: Path

    def __post_init__(self):
        self.meter_weights_path = Path(self.meter_weights_path)
        self.diacritizer_weights_path = Path(self.diacritizer_weights_path)
        
        if not self.meter_weights_path.exists():
            raise FileNotFoundError(f"Meter weights not found at: {self.meter_weights_path}")
        if not self.diacritizer_weights_path.exists():
            raise FileNotFoundError(f"Diacritizer weights not found at: {self.diacritizer_weights_path}")

@dataclass
class OperationConfig:
    poem_generator: LLMConfig
    poem_evaluator: LLMConfig
    query_transform: LLMConfig
    verse_reviewer: LLMConfig
    verse_reviser: LLMConfig

@dataclass
class Config:
    task: dict
    model_paths: ModelPaths
    operations: OperationConfig
    db_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.db_path:
            self.db_path = Path(self.db_path)
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found at: {self.db_path}")
            
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_paths = ModelPaths(**config_dict.get('model_paths', {}))
        
        operations = {}
        for op in ['poem_generator', 'poem_evaluator', 'query_transform', 
                  'verse_reviewer', 'verse_reviser']:
            if op in config_dict.get('operations', {}):
                operations[op] = LLMConfig(**config_dict['operations'][op])
        
        operations_config = OperationConfig(**operations)
        
        return cls(
            task=config_dict.get('task', {}),
            model_paths=model_paths,
            operations=operations_config,
            db_path=config_dict.get('db_path'),
        )