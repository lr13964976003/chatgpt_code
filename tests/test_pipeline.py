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


def test_chakra_execution_trace_without_ts_dur(tmp_path: Path):
    execution = {
        "schema": "1.1.1-chakra.0.0.4",
        "start_ts": 197820970,
        "nodes": [
            {
                "id": 2,
                "name": "[pytorch|profiler|execution_trace|thread]",
                "attrs": [{"name": "tid", "value": 3}],
                "inputs": {"shapes": [], "types": []},
            },
            {
                "id": 3,
                "name": "aten::mul",
                "attrs": [{"name": "tid", "value": 8}, {"name": "scope", "value": 7}],
                "inputs": {"shapes": [[4, 4], [4, 4]], "types": ["Float", "Float"]},
            },
        ],
    }
    kineto = {
        "traceEvents": [
            {
                "name": "aten::mul",
                "ts": 10.0,
                "dur": 12.0,
                "stream": 4,
                "ph": "X",
                "args": {"Input Dims": [[4, 4], [4, 4]], "Input type": ["float32", "float32"]},
            }
        ]
    }

    exe_path = tmp_path / "execution_chakra.json"
    kin_path = tmp_path / "kineto.json"
    exe_path.write_text(json.dumps(execution), encoding="utf-8")
    kin_path.write_text(json.dumps(kineto), encoding="utf-8")

    exe_events = load_execution_trace(exe_path)
    assert len(exe_events) == 1
    assert exe_events[0].name == "aten::mul"
    assert exe_events[0].duration_us == 0.0
    assert exe_events[0].ts_us >= 197820970

    workload = merge_execution_and_kineto(exe_events, load_kineto_trace(kin_path), "deepseekv3")
    event = workload.events[0]

    assert event.duration_us == 12.0
    assert event.stream == 4
    assert len(event.inputs) == 2
