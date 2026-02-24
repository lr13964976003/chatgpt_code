from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import OpEvent, TensorSpec


class TraceFormatError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_list(data: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "traceEvents" in data and isinstance(data["traceEvents"], list):
        return data["traceEvents"]
    raise TraceFormatError(f"无法识别的 trace 格式: {path}")


def load_execution_trace(path: str | Path) -> List[OpEvent]:
    """
    加载 deepseekv3 运行时 execution trace。

    支持常见格式:
    - [{"name": "aten::mm", "ts": 1.2, "dur": 3.4, ...}, ...]
    - {"events": [...]} / {"traceEvents": [...]} 兼容处理
    """
    file_path = Path(path)
    data = _read_json(file_path)

    raw_events: Iterable[Dict[str, Any]]
    if isinstance(data, dict) and "events" in data and isinstance(data["events"], list):
        raw_events = data["events"]
    else:
        raw_events = _ensure_list(data, file_path)

    events: List[OpEvent] = []
    for item in raw_events:
        name = item.get("name") or item.get("op")
        if not name:
            continue

        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        shapes = args.get("input_shapes") or item.get("input_shapes") or []
        dtypes = args.get("input_dtypes") or item.get("input_dtypes") or []

        tensors: List[TensorSpec] = []
        for i, shape in enumerate(shapes):
            if not isinstance(shape, list):
                continue
            dtype = dtypes[i] if i < len(dtypes) else "float16"
            tensors.append(TensorSpec(name=f"{name}_in{i}", shape=shape, dtype=dtype))

        events.append(
            OpEvent(
                name=name,
                ts_us=float(item.get("ts", item.get("timestamp", 0.0))),
                duration_us=float(item.get("dur", item.get("duration", 0.0))),
                stream=item.get("stream") or args.get("stream"),
                thread_id=item.get("tid"),
                category=item.get("cat", "execution"),
                inputs=tensors,
                attrs=args,
            )
        )
    return events


def load_kineto_trace(path: str | Path) -> List[OpEvent]:
    """
    加载 kineto trace (chrome trace json)。
    过滤出算子级事件并做统一结构转换。
    """
    file_path = Path(path)
    raw_events = _ensure_list(_read_json(file_path), file_path)
    events: List[OpEvent] = []

    for item in raw_events:
        if item.get("ph") not in ("X", "B", "E", None):
            continue

        name = item.get("name", "")
        if not name or "ProfilerStep" in name:
            continue

        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        input_shapes = args.get("Input Dims") or args.get("input_shapes") or []
        input_types = args.get("Input type") or args.get("input_dtypes") or []

        tensors: List[TensorSpec] = []
        if isinstance(input_shapes, list):
            for i, shape in enumerate(input_shapes):
                if isinstance(shape, list):
                    dtype = (
                        input_types[i]
                        if isinstance(input_types, list) and i < len(input_types)
                        else "float16"
                    )
                    tensors.append(TensorSpec(name=f"{name}_in{i}", shape=shape, dtype=str(dtype)))

        events.append(
            OpEvent(
                name=name,
                ts_us=float(item.get("ts", 0.0)),
                duration_us=float(item.get("dur", 0.0)),
                stream=item.get("stream") or args.get("stream"),
                thread_id=item.get("tid"),
                category=item.get("cat", "kineto"),
                inputs=tensors,
                attrs=args,
            )
        )

    return events
