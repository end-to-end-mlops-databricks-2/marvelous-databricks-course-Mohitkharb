"""
This module contains the configuration for the power consumption project.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml

class ModelConfig(BaseModel):
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    type: str
    test_size: float = 0.2
    params: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class DataConfig(BaseModel):
    workspace_location: Optional[str] = None
    user_name: Optional[str] = None
    format: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class ProjectConfig(BaseModel):
    catalog_name: str
    schema_name: str
    use_case_name: str
    project_name: str 
    branch_name: Optional[str] = None
    ml_model_settings: Optional[ModelConfig] = None
    data_settings: Optional[DataConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ProjectConfig":
        """
        Create a ProjectConfig instance from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            ProjectConfig: An instance of ProjectConfig with the loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValidationError: If the YAML content doesn't match the expected schema
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)



