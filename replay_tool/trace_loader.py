from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _attrs_list_to_dict(attrs: Any) -> Dict[str, Any]:
    if not isinstance(attrs, list):
        return {}
    result: Dict[str, Any] = {}
    for item in attrs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        result[str(name)] = item.get("value")
    return result


def _extract_inputs_from_node(name: str, node: Dict[str, Any]) -> List[TensorSpec]:
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    shapes = inputs.get("shapes") if isinstance(inputs.get("shapes"), list) else []
    types = inputs.get("types") if isinstance(inputs.get("types"), list) else []

    tensors: List[TensorSpec] = []
    for i, shape in enumerate(shapes):
        if not isinstance(shape, list):
            continue
        if not shape:
            continue
        dtype = str(types[i]).lower() if i < len(types) else "float16"
        tensors.append(TensorSpec(name=f"{name}_in{i}", shape=shape, dtype=dtype))
    return tensors


def _normalize_execution_events(data: Any) -> Tuple[List[Dict[str, Any]], float]:
    """返回 events 列表和可选的 start_ts。"""
    start_ts = 0.0
    if isinstance(data, dict):
        if "start_ts" in data:
            try:
                start_ts = float(data.get("start_ts", 0.0))
            except (TypeError, ValueError):
                start_ts = 0.0
        if "events" in data and isinstance(data["events"], list):
            return data["events"], start_ts
        if "nodes" in data and isinstance(data["nodes"], list):
            return data["nodes"], start_ts
    return _ensure_list(data, Path("<execution_trace>")), start_ts


def load_execution_trace(path: str | Path) -> List[OpEvent]:
    """
    加载 deepseekv3 运行时 execution trace。

    支持格式:
    - [{"name": "aten::mm", "ts": 1.2, "dur": 3.4, ...}, ...]
    - {"events": [...]} / {"traceEvents": [...]}。
    - Chakra execution trace: {"schema": "...", "start_ts": ..., "nodes": [...]}。
    """
    file_path = Path(path)
    data = _read_json(file_path)
    raw_events, start_ts = _normalize_execution_events(data)

    events: List[OpEvent] = []
    for idx, item in enumerate(raw_events):
        name = item.get("name") or item.get("op")
        if not name:
            continue

        # 过滤框架元事件，避免把线程/进程组初始化当成算子重放。
        if str(name).startswith("[pytorch|") or str(name).startswith("## process_group:init ##"):
            continue

        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        attrs_from_list = _attrs_list_to_dict(item.get("attrs"))
        merged_attrs = {**attrs_from_list, **args}

        # 标准 execution trace (events)
        shapes = merged_attrs.get("input_shapes") or item.get("input_shapes") or []
        dtypes = merged_attrs.get("input_dtypes") or item.get("input_dtypes") or []

        tensors: List[TensorSpec] = []
        for i, shape in enumerate(shapes):
            if not isinstance(shape, list):
                continue
            dtype = dtypes[i] if i < len(dtypes) else "float16"
            tensors.append(TensorSpec(name=f"{name}_in{i}", shape=shape, dtype=str(dtype)))

        # Chakra nodes 里用 inputs.shapes/types 表示输入
        if not tensors and "inputs" in item:
            tensors = _extract_inputs_from_node(str(name), item)

        tid = item.get("tid")
        if tid is None:
            tid = attrs_from_list.get("tid")

        stream = item.get("stream") or merged_attrs.get("stream")

        # execution 可能没有 ts/dur，尝试用 start_ts + 序号生成稳定时间轴
        ts_raw = item.get("ts", item.get("timestamp"))
        if ts_raw is None:
            ts_raw = start_ts + idx

        dur_raw = item.get("dur", item.get("duration"))
        if dur_raw is None:
            dur_raw = 0.0

        events.append(
            OpEvent(
                name=str(name),
                ts_us=float(ts_raw),
                duration_us=float(dur_raw),
                stream=stream,
                thread_id=int(tid) if isinstance(tid, (int, float)) else None,
                category=item.get("cat", "execution"),
                inputs=tensors,
                attrs=merged_attrs,
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
