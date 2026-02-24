from __future__ import annotations

import argparse
from pathlib import Path

from .codegen import dump_workload_json, generate_python_replay_script
from .trace_loader import load_execution_trace, load_kineto_trace
from .workload_builder import merge_execution_and_kineto


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 deepseekv3 execution trace + kineto trace 生成 Ascend NPU 可运行负载程序"
    )
    parser.add_argument("--execution-trace", required=True, help="execution trace json 文件路径")
    parser.add_argument("--kineto-trace", required=True, help="kineto trace json 文件路径")
    parser.add_argument("--model-name", default="deepseekv3", help="模型名称")
    parser.add_argument("--out-dir", default="generated", help="输出目录")
    parser.add_argument("--script-name", default="replay_workload.py", help="生成脚本文件名")
    parser.add_argument("--workload-name", default="workload.json", help="中间工作负载 json 文件名")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_events = load_execution_trace(args.execution_trace)
    kineto_events = load_kineto_trace(args.kineto_trace)

    workload = merge_execution_and_kineto(execution_events, kineto_events, args.model_name)

    workload_path = out_dir / args.workload_name
    script_path = out_dir / args.script_name

    dump_workload_json(workload, workload_path)
    generate_python_replay_script(workload, script_path)

    print(f"[OK] workload json: {workload_path}")
    print(f"[OK] replay script: {script_path}")
    print(f"[INFO] event count: {len(workload.events)}")


if __name__ == "__main__":
    main()
