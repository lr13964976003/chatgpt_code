# DeepSeekV3 Ascend NPU Trace Replay Generator

该项目提供一套可以直接落地的工具链：

1. 读取 `execution trace`（运行时算子事件）。
2. 读取 `kineto trace`（Chrome trace / profiler trace）。
3. 自动融合两类 trace 信息，生成统一 `workload.json`。
4. 自动导出可执行 Python 负载脚本，在昇腾 NPU (`torch_npu`) 上重放模型执行过程。

> 目标：尽可能重现 DeepSeekV3 在真实运行时中的算子序列、输入形状和时间行为。

## 目录结构

- `generate_replay.py`：主入口。
- `replay_tool/trace_loader.py`：trace 解析。
- `replay_tool/workload_builder.py`：execution + kineto 融合。
- `replay_tool/codegen.py`：生成 workload JSON 和 replay 脚本。

## 快速开始

```bash
python generate_replay.py \
  --execution-trace /path/to/execution_trace.json \
  --kineto-trace /path/to/kineto_trace.json \
  --model-name deepseekv3 \
  --out-dir generated
```

执行后会生成：

- `generated/workload.json`
- `generated/replay_workload.py`

在昇腾机器上运行重放：

```bash
python generated/replay_workload.py --warmup 2 --iters 10 --device npu:0
```

## 支持的 Trace 格式

### execution trace

支持以下常见格式：

- `[{...}, {...}]`
- `{"events": [{...}]}`
- `{"traceEvents": [{...}]}`
- Chakra execution trace: `{"schema": "1.1.1-chakra.*", "start_ts": ..., "nodes": [...]}`（即使没有 ts/dur 也支持）

关键字段示例（events 模式）：

```json
{
  "name": "aten::matmul",
  "ts": 123.4,
  "dur": 56.7,
  "stream": 7,
  "args": {
    "input_shapes": [[4096, 4096], [4096, 4096]],
    "input_dtypes": ["float16", "float16"]
  }
}
```


Chakra `nodes` 模式示例（无 `ts/dur` 也可）：

```json
{
  "schema": "1.1.1-chakra.0.0.4",
  "start_ts": 197820970,
  "nodes": [
    {
      "id": 3,
      "name": "aten::mul",
      "inputs": {"shapes": [[4, 4], [4, 4]], "types": ["Float", "Float"]},
      "attrs": [{"name": "tid", "value": 8}]
    }
  ]
}
```

该模式下工具会：
- 从 `inputs.shapes/types` 推断输入张量；
- 从 `attrs` 提取 `tid` 等信息；
- 若节点没有 `ts/dur`，使用 `start_ts + 节点序号` 生成稳定顺序，并优先用 kineto 的 `dur/stream` 进行补全。

### kineto trace

支持 chrome trace 常见结构：

- 顶层 `traceEvents` 列表
- `ph` in `X/B/E/None`
- 从 `args["Input Dims"]` 与 `args["Input type"]` 提取输入信息

## 设计说明

- **融合策略**：以 execution trace 为主序，按 `(op_name, stream)` 关联 kineto 事件。
- **补全能力**：当 execution 缺失 duration/stream/inputs 时，用 kineto 进行补齐。
- **算子映射**：内置 `matmul/addmm/layer_norm/silu/relu/add/mul` 映射；未知算子自动 fallback 为 `clone + sleep`。
- **可执行性**：生成脚本会直接创建 NPU 张量并执行算子序列，最后输出每次迭代耗时。

## 注意事项

1. 需要在目标环境安装 `torch` + `torch_npu`。
2. trace 中没有输入 shape 时，会退化为最小可运行路径（默认 shape）。
3. 若需要更高保真（如通信算子、流水并行 stage、KV cache 时序），建议在 `codegen.py` 中扩展 `_emit_op` 规则。
