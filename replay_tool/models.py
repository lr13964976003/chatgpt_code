from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TensorSpec:
    """描述重放中需要构造的张量信息。"""

    name: str
    shape: List[int]
    dtype: str = "float16"
    device: str = "npu"


@dataclass
class OpEvent:
    """统一后的算子事件。"""

    name: str
    ts_us: float
    duration_us: float
    stream: Optional[int]
    thread_id: Optional[int]
    category: str
    inputs: List[TensorSpec] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayWorkload:
    """用于生成重放脚本的高层工作负载结构。"""

    model_name: str
    events: List[OpEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)
