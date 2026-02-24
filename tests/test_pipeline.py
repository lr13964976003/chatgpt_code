import json
from pathlib import Path

from replay_tool.codegen import dump_workload_json, generate_python_replay_script
from replay_tool.trace_loader import load_execution_trace, load_kineto_trace
from replay_tool.workload_builder import merge_execution_and_kineto


def test_end_to_end_generation(tmp_path: Path):
    execution = [
        {
            "name": "aten::matmul",
            "ts": 1.0,
            "dur": 0.0,
            "stream": 1,
            "args": {"input_shapes": [[2, 2], [2, 2]], "input_dtypes": ["float16", "float16"]},
        },
        {
            "name": "aten::add",
            "ts": 3.0,
            "dur": 2.0,
            "stream": 1,
            "args": {"input_shapes": [[2, 2], [2, 2]], "input_dtypes": ["float16", "float16"]},
        },
    ]
    kineto = {
        "traceEvents": [
            {
                "name": "aten::matmul",
                "ts": 1.2,
                "dur": 5.0,
                "stream": 1,
                "ph": "X",
                "args": {"Input Dims": [[2, 2], [2, 2]], "Input type": ["float16", "float16"]},
            }
        ]
    }

    exe_path = tmp_path / "execution.json"
    kin_path = tmp_path / "kineto.json"
    exe_path.write_text(json.dumps(execution), encoding="utf-8")
    kin_path.write_text(json.dumps(kineto), encoding="utf-8")

    exe_events = load_execution_trace(exe_path)
    kin_events = load_kineto_trace(kin_path)
    workload = merge_execution_and_kineto(exe_events, kin_events, "deepseekv3")

    workload_path = tmp_path / "workload.json"
    script_path = tmp_path / "replay.py"
    dump_workload_json(workload, workload_path)
    generate_python_replay_script(workload, script_path)

    data = json.loads(workload_path.read_text(encoding="utf-8"))
    script = script_path.read_text(encoding="utf-8")

    assert data["metadata"]["merged_event_count"] == 2
    assert "aten::matmul" in script
    assert "torch.matmul" in script
