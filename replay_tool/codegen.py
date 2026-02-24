from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import OpEvent, ReplayWorkload, TensorSpec


def _dtype_expr(dtype: str) -> str:
    normalized = dtype.lower().replace("torch.", "")
    mapping = {
        "float16": "torch.float16",
        "half": "torch.float16",
        "float32": "torch.float32",
        "float": "torch.float32",
        "bfloat16": "torch.bfloat16",
        "int64": "torch.int64",
        "int32": "torch.int32",
    }
    return mapping.get(normalized, "torch.float16")


def _emit_tensor_builders(tensors: List[TensorSpec]) -> List[str]:
    lines: List[str] = []
    for t in tensors:
        shape = t.shape if t.shape else [1]
        lines.append(
            f"    tensor_cache['{t.name}'] = torch.randn({shape}, dtype={_dtype_expr(t.dtype)}, device=device)"
        )
    return lines


def _emit_op(op: OpEvent) -> str:
    op_name = op.name
    inputs = [f"tensor_cache['{t.name}']" for t in op.inputs]
    duration = max(op.duration_us, 0.0)

    if "mm" in op_name or "matmul" in op_name:
        if len(inputs) >= 2:
            return f"    out = torch.matmul({inputs[0]}, {inputs[1]})"
    if "addmm" in op_name and len(inputs) >= 3:
        return f"    out = torch.addmm({inputs[0]}, {inputs[1]}, {inputs[2]})"
    if "layer_norm" in op_name and inputs:
        return (
            "    out = torch.nn.functional.layer_norm("
            f"{inputs[0]}, normalized_shape={tuple(op.inputs[0].shape[-1:])})"
        )
    if "silu" in op_name and inputs:
        return f"    out = torch.nn.functional.silu({inputs[0]})"
    if "relu" in op_name and inputs:
        return f"    out = torch.relu({inputs[0]})"
    if "mul" in op_name and len(inputs) >= 2:
        return f"    out = {inputs[0]} * {inputs[1]}"
    if "add" in op_name and len(inputs) >= 2:
        return f"    out = {inputs[0]} + {inputs[1]}"

    if inputs:
        sleep_s = duration / 1_000_000
        return (
            "    # 未映射算子: 使用 clone + sleep 模拟时延\n"
            f"    out = {inputs[0]}.clone()\n"
            f"    time.sleep({sleep_s:.8f})"
        )
    return "    out = torch.empty((1,), device=device)"


def generate_python_replay_script(workload: ReplayWorkload, out_path: str | Path) -> None:
    out_file = Path(out_path)

    # 去重张量声明
    tensor_map: Dict[str, TensorSpec] = {}
    for event in workload.events:
        for tensor in event.inputs:
            tensor_map.setdefault(tensor.name, tensor)

    build_lines = _emit_tensor_builders(list(tensor_map.values()))

    op_lines: List[str] = []
    for i, event in enumerate(workload.events):
        op_lines.append(f"    # [{i}] {event.name} stream={event.stream} dur_us={event.duration_us}")
        op_lines.extend(_emit_op(event).split("\n"))
        op_lines.append(f"    tensor_cache['event_{i}_out'] = out")

    script = f'''#!/usr/bin/env python3
import argparse
import json
import time

import torch

try:
    import torch_npu
except Exception as e:
    raise RuntimeError("请在昇腾环境安装 torch_npu 后运行") from e


def main():
    parser = argparse.ArgumentParser(description="Replay workload generated from execution+kineto traces")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    tensor_cache = {{}}

{chr(10).join(build_lines) if build_lines else '    # no tensors inferred'}

    for _ in range(args.warmup):
        for _op in range(1):
            pass
        if torch.npu.is_available():
            torch.npu.synchronize()

    latencies_ms = []
    for _ in range(args.iters):
        st = time.perf_counter()
{chr(10).join(op_lines) if op_lines else '        pass'}
        if torch.npu.is_available():
            torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - st) * 1000)

    result = {{
        "model_name": {json.dumps(workload.model_name)},
        "iters": args.iters,
        "latencies_ms": latencies_ms,
        "avg_ms": sum(latencies_ms) / max(1, len(latencies_ms)),
        "event_count": {len(workload.events)},
    }}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''

    out_file.write_text(script, encoding="utf-8")
    out_file.chmod(0o755)


def dump_workload_json(workload: ReplayWorkload, out_path: str | Path) -> None:
    out_file = Path(out_path)
    data = {
        "model_name": workload.model_name,
        "metadata": workload.metadata,
        "events": [
            {
                "name": e.name,
                "ts_us": e.ts_us,
                "duration_us": e.duration_us,
                "stream": e.stream,
                "thread_id": e.thread_id,
                "category": e.category,
                "inputs": [
                    {
                        "name": t.name,
                        "shape": t.shape,
                        "dtype": t.dtype,
                        "device": t.device,
                    }
                    for t in e.inputs
                ],
                "attrs": e.attrs,
            }
            for e in workload.events
        ],
    }
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
