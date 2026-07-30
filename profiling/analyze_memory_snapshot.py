"""Inspect the largest real CUDA allocation recorded in a PyTorch snapshot.

The visualizer's rectangles are useful for temporal context, but their width
and height are not a reliable way to determine an allocation's byte size.
This CLI instead reads ``device_traces[*]`` and computes the maximum over
events whose action is exactly ``"alloc"``.  It keeps tied maxima separate:
one workload can have many largest allocations with different call stacks.

Only load snapshots produced by a trusted PyTorch run.  PyTorch writes memory
snapshots with ``pickle``, which is not safe to deserialize from an untrusted
source.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MemorySnapshotAnalysisError(ValueError):
    """Raised when a file does not have the expected PyTorch snapshot shape."""


@dataclass(frozen=True)
class AllocationEvent:
    """One allocator ``alloc`` event from a device trace."""

    device_index: int
    event_index: int
    size_bytes: int
    address: int | None
    stream: int | None
    time_us: int | float | None
    frames: tuple[dict[str, str | int], ...]


@dataclass(frozen=True)
class TiedTraceGroup:
    """Equivalent full stack traces among allocations of the largest size."""

    events: tuple[AllocationEvent, ...]

    @property
    def representative(self) -> AllocationEvent:
        """Return the earliest occurrence of this exact trace."""

        return self.events[0]


@dataclass(frozen=True)
class SnapshotAnalysis:
    """The allocation facts extracted from one snapshot."""

    allocation_event_count: int
    largest_allocation_bytes: int
    tied_events: tuple[AllocationEvent, ...]
    tied_trace_groups: tuple[TiedTraceGroup, ...]

    @property
    def representative(self) -> AllocationEvent:
        """Return the earliest event among the tied maximum allocations."""

        return self.tied_events[0]


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MemorySnapshotAnalysisError(f"{context} must be a non-negative integer, got {value!r}.")
    return value


def _optional_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, context=context)


def _optional_time(value: Any, *, context: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemorySnapshotAnalysisError(f"{context} must be a number or null, got {value!r}.")
    return value


def _frames(value: Any, *, context: str) -> tuple[dict[str, str | int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MemorySnapshotAnalysisError(f"{context} must be a list of stack frames.")

    frames: list[dict[str, str | int]] = []
    for frame_index, frame in enumerate(value):
        if not isinstance(frame, Mapping):
            raise MemorySnapshotAnalysisError(f"{context}[{frame_index}] must be an object.")
        name = frame.get("name", "")
        filename = frame.get("filename", "")
        line = frame.get("line", 0)
        if not isinstance(name, str) or not isinstance(filename, str):
            raise MemorySnapshotAnalysisError(f"{context}[{frame_index}] has a non-string name or filename.")
        if isinstance(line, bool) or not isinstance(line, int):
            raise MemorySnapshotAnalysisError(f"{context}[{frame_index}].line must be an integer.")
        frames.append({"name": name, "filename": filename, "line": line})
    return tuple(frames)


def load_snapshot(path: Path) -> Mapping[str, Any]:
    """Load and minimally validate a trusted PyTorch memory-history snapshot."""

    try:
        with path.open("rb") as snapshot_file:
            snapshot = pickle.load(snapshot_file)
    except FileNotFoundError as error:
        raise MemorySnapshotAnalysisError(f"Snapshot does not exist: {path}") from error
    except (pickle.UnpicklingError, EOFError) as error:
        raise MemorySnapshotAnalysisError(f"Cannot read PyTorch memory snapshot {path}: {error}") from error

    if not isinstance(snapshot, Mapping):
        raise MemorySnapshotAnalysisError(f"{path}: snapshot root must be an object.")
    if not isinstance(snapshot.get("device_traces"), Sequence):
        raise MemorySnapshotAnalysisError(f"{path}: snapshot has no device_traces list.")
    return snapshot


def allocation_events(snapshot: Mapping[str, Any]) -> list[AllocationEvent]:
    """Return only true ``alloc`` events, never segment or free records."""

    traces = snapshot["device_traces"]
    assert isinstance(traces, Sequence)
    events: list[AllocationEvent] = []

    for device_index, trace in enumerate(traces):
        if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes, bytearray)):
            raise MemorySnapshotAnalysisError(f"device_traces[{device_index}] must be a list of events.")
        for event_index, event in enumerate(trace):
            if not isinstance(event, Mapping):
                raise MemorySnapshotAnalysisError(f"device_traces[{device_index}][{event_index}] must be an object.")
            if event.get("action") != "alloc":
                continue
            context = f"device_traces[{device_index}][{event_index}]"
            events.append(
                AllocationEvent(
                    device_index=device_index,
                    event_index=event_index,
                    size_bytes=_nonnegative_int(event.get("size"), context=f"{context}.size"),
                    address=_optional_int(event.get("addr"), context=f"{context}.addr"),
                    stream=_optional_int(event.get("stream"), context=f"{context}.stream"),
                    time_us=_optional_time(event.get("time_us"), context=f"{context}.time_us"),
                    frames=_frames(event.get("frames", []), context=f"{context}.frames"),
                )
            )
    return events


def _trace_key(event: AllocationEvent) -> tuple[tuple[str, str, int], ...]:
    return tuple((str(frame["name"]), str(frame["filename"]), int(frame["line"])) for frame in event.frames)


def analyze_snapshot(snapshot: Mapping[str, Any]) -> SnapshotAnalysis:
    """Find the maximum real allocation and group every event tied for it."""

    events = allocation_events(snapshot)
    if not events:
        raise MemorySnapshotAnalysisError("Snapshot contains no allocator events with action == 'alloc'.")

    largest_allocation_bytes = max(event.size_bytes for event in events)
    tied_events = tuple(event for event in events if event.size_bytes == largest_allocation_bytes)
    groups_by_trace: dict[tuple[tuple[str, str, int], ...], list[AllocationEvent]] = defaultdict(list)
    for event in tied_events:
        groups_by_trace[_trace_key(event)].append(event)

    groups = [TiedTraceGroup(events=tuple(group_events)) for group_events in groups_by_trace.values()]
    groups.sort(key=lambda group: (-len(group.events), group.representative.device_index, group.representative.event_index))
    return SnapshotAnalysis(
        allocation_event_count=len(events),
        largest_allocation_bytes=largest_allocation_bytes,
        tied_events=tied_events,
        tied_trace_groups=tuple(groups),
    )


def _format_size(size_bytes: int) -> str:
    return f"{size_bytes:,} bytes ({size_bytes / 2**20:.3f} MiB, {size_bytes / 2**30:.3f} GiB)"


def _event_location(event: AllocationEvent) -> str:
    return f"device {event.device_index}, event {event.event_index}"


def _application_callsite(event: AllocationEvent) -> str:
    """Return the nearest non-package Python callsite, when captured."""

    python_frames = [frame for frame in event.frames if str(frame["filename"]).endswith(".py")]
    source_frames = [frame for frame in python_frames if "site-packages" not in str(frame["filename"])]
    frame = source_frames[0] if source_frames else (python_frames[0] if python_frames else None)
    if frame is None:
        return "no Python source frame captured"
    return f"{frame['filename']}:{frame['line']} ({frame['name']})"


def event_as_dict(event: AllocationEvent) -> dict[str, Any]:
    """Turn an event into JSON-safe data without retaining unrelated snapshot data."""

    return {
        "device_index": event.device_index,
        "event_index": event.event_index,
        "size_bytes": event.size_bytes,
        "size_mib": event.size_bytes / 2**20,
        "address": event.address,
        "stream": event.stream,
        "time_us": event.time_us,
        "application_callsite": _application_callsite(event),
        "frames": list(event.frames),
    }


def analysis_as_dict(snapshot_path: Path, analysis: SnapshotAnalysis) -> dict[str, Any]:
    """Produce the complete, reproducible report representation."""

    return {
        "schema_version": 1,
        "snapshot": str(snapshot_path),
        "allocation_event_count": analysis.allocation_event_count,
        "largest_allocation_bytes": analysis.largest_allocation_bytes,
        "largest_allocation_mib": analysis.largest_allocation_bytes / 2**20,
        "largest_allocation_gib": analysis.largest_allocation_bytes / 2**30,
        "tied_largest_allocation_events": len(analysis.tied_events),
        "distinct_tied_stack_traces": len(analysis.tied_trace_groups),
        "representative_event": event_as_dict(analysis.representative),
        "tied_trace_groups": [
            {
                "event_count": len(group.events),
                "representative_event": event_as_dict(group.representative),
            }
            for group in analysis.tied_trace_groups
        ],
    }


def format_text_report(snapshot_path: Path, analysis: SnapshotAnalysis, *, show_frames: bool, max_trace_groups: int) -> str:
    """Format a human-readable report, including the full representative trace."""

    representative = analysis.representative
    lines = [
        f"Snapshot: {snapshot_path}",
        f"Real alloc events: {analysis.allocation_event_count:,}",
        f"Largest single alloc: {_format_size(analysis.largest_allocation_bytes)}",
        f"Tied largest alloc events: {len(analysis.tied_events):,}",
        f"Distinct full stack traces among ties: {len(analysis.tied_trace_groups):,}",
        "",
        "Representative largest allocation (earliest tied event):",
        f"  location: {_event_location(representative)}",
        f"  address: {representative.address}",
        f"  stream: {representative.stream}",
        f"  time_us: {representative.time_us}",
        f"  application callsite: {_application_callsite(representative)}",
        f"  captured stack frames: {len(representative.frames)}",
    ]
    if show_frames:
        lines.extend(["", "Full stack trace (allocator to Python caller):"])
        for frame_index, frame in enumerate(representative.frames):
            lines.append(f"  #{frame_index:02d} {frame['name']}  ({frame['filename']}:{frame['line']})")

    lines.extend(["", "Tied maximum stack-trace groups:"])
    for group_index, group in enumerate(analysis.tied_trace_groups[:max_trace_groups], start=1):
        event = group.representative
        lines.append(f"  {group_index}. {len(group.events):,} event(s); {_event_location(event)}; {_application_callsite(event)}")
    omitted_groups = len(analysis.tied_trace_groups) - max_trace_groups
    if omitted_groups > 0:
        lines.append(f"  ... {omitted_groups:,} additional group(s) are present; use --json-output to export every full stack trace.")
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find the largest real CUDA allocation in a trusted PyTorch memory snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to a PyTorch memory-history .pickle snapshot.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON report path. Includes every tied maximum event group's full representative stack trace.",
    )
    parser.add_argument("--no-frames", action="store_true", help="Do not print the full stack trace for the representative allocation.")
    parser.add_argument(
        "--max-trace-groups",
        type=_positive_int,
        default=16,
        help="Maximum number of tied stack-trace groups to show in text output (default: 16).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        snapshot = load_snapshot(args.snapshot)
        analysis = analyze_snapshot(snapshot)
    except MemorySnapshotAnalysisError as error:
        raise SystemExit(f"error: {error}") from error

    print(format_text_report(args.snapshot, analysis, show_frames=not args.no_frames, max_trace_groups=args.max_trace_groups))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(analysis_as_dict(args.snapshot, analysis), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON report: {args.json_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
