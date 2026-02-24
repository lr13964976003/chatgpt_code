from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

from .models import OpEvent, ReplayWorkload


def _event_key(event: OpEvent) -> Tuple[str, int | None]:
    return event.name, event.stream


def merge_execution_and_kineto(
    execution_events: List[OpEvent], kineto_events: List[OpEvent], model_name: str
) -> ReplayWorkload:
    """
    合并两类 trace:
    - 以 execution trace 为主体顺序。
    - 用 kineto 的 duration / stream / 低层信息补全。
    - 当 execution 缺失 stream 时，回退到仅按 op_name 匹配。
    """
    kineto_index: Dict[Tuple[str, int | None], List[OpEvent]] = {}
    kineto_name_only: Dict[str, List[OpEvent]] = {}
    for event in kineto_events:
        kineto_index.setdefault(_event_key(event), []).append(event)
        kineto_name_only.setdefault(event.name, []).append(event)

    merged: List[OpEvent] = []
    for exe in sorted(execution_events, key=lambda x: x.ts_us):
        candidates = kineto_index.get(_event_key(exe), [])
        kin: OpEvent | None = None
        if candidates:
            kin = candidates.pop(0)
        elif exe.stream is None:
            fallback = kineto_name_only.get(exe.name, [])
            if fallback:
                kin = fallback.pop(0)

        if kin is not None:
            if exe.duration_us <= 0 and kin.duration_us > 0:
                exe.duration_us = kin.duration_us
            if exe.stream is None:
                exe.stream = kin.stream
            exe.attrs = {**kin.attrs, **exe.attrs}
            if not exe.inputs and kin.inputs:
                exe.inputs = kin.inputs

        merged.append(exe)

    return ReplayWorkload(
        model_name=model_name,
        events=merged,
        metadata={
            "execution_event_count": len(execution_events),
            "kineto_event_count": len(kineto_events),
            "merged_event_count": len(merged),
        },
    )


def workload_to_dict(workload: ReplayWorkload) -> dict:
    return {
        "model_name": workload.model_name,
        "metadata": workload.metadata,
        "events": [asdict(event) for event in workload.events],
    }
