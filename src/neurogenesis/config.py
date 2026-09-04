"""Configuration and provenance.

Frozen dataclasses plus a small YAML loader, deliberately *not* Hydra. The sweep
axis in this project is generated tasks, which is programmatic, so an override
grammar buys nothing; meanwhile Hydra changes the working directory per run
(which fights an append-only results store) and its ``DictConfig`` does not hash
stably into a run id. What is actually needed is three things -- load, hash,
skip-if-already-run -- and that is what this module is.

The config hash is the run identity: re-running an identical config is a no-op,
so a long sweep resumes safely after any interruption.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskSpec:
    """How to build the task. ``seed`` also enforces the dev/confirmatory split.

    ``family`` selects a generator:
      ``algebraic`` -- ``y = (sum w_i c_i) mod m``, closed-form ``|RS|``
      ``addition``  -- ``y = sum c_i`` over the integers
      ``planted``   -- a chosen symmetry monoid, ``RS`` measured not assumed
      ``random``    -- random total label table (null family, no planted structure)
      ``rarefied``  -- identifiable with a deliberately tiny margin (the H2 cell)
    """

    family: str = "algebraic"
    weights: tuple[int, ...] = (1, 1)
    m: int = 10
    k: int | None = None
    n_slots: int = 2
    support_density: float = 1.0
    planted_kind: str = "cyclic"
    rarity: float = 1.0
    swap: tuple[int, int] = (0, 1)
    seed: int = 0


@dataclass(frozen=True)
class DataSpec:
    tier: str = "M"  # "M" = MNIST digits, "S" = synthetic vectors
    n_train: int = 25_000
    n_val: int = 5_000
    n_test: int = 5_000
    noise: float = 0.0  # Tier S only
    dim: int = 32  # Tier S only
    data_seed: int = 0


@dataclass(frozen=True)
class ModelSpec:
    encoder: str = "cnn"
    init_seed: int = 0


@dataclass(frozen=True)
class OptimSpec:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    lr_final: float = 1e-4
    weight_decay: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    """One experimental run: task, data, model, optimiser."""

    experiment: str = "e1"
    task: TaskSpec = field(default_factory=TaskSpec)
    data: DataSpec = field(default_factory=DataSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    device: str = "cpu"
    num_threads: int = 1
    tuning_mode: bool = False

    def __post_init__(self) -> None:
        # Leakage control, enforced mechanically rather than by good intentions.
        if self.tuning_mode and self.task.seed >= CONFIRMATORY_SEED_FLOOR:
            raise ValueError(
                f"task.seed={self.task.seed} is in the confirmatory range "
                f"(>= {CONFIRMATORY_SEED_FLOOR}) and must never be used for tuning. "
                "Tune on dev tasks (seed < 100) only."
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))

    def config_hash(self) -> str:
        """Stable sha256 over the canonical JSON -- the run's identity."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


#: Task-generator seeds at or above this are *confirmatory*: they are run exactly
#: once with the frozen recipe and are never touched during tuning.
CONFIRMATORY_SEED_FLOOR = 1000
#: Seeds below this are *dev* tasks -- the only ones any hyperparameter may see.
DEV_SEED_CEILING = 100


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_dict(cls, data: dict):
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(f.type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str | Path, **overrides: Any) -> RunConfig:
    """Load a YAML config, applying dotted-key overrides like ``task.m=7``."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key, val in overrides.items():
        target = data
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = val
    nested = {
        "task": TaskSpec,
        "data": DataSpec,
        "model": ModelSpec,
        "optim": OptimSpec,
    }
    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        if k in nested and isinstance(v, dict):
            kwargs[k] = _from_dict(nested[k], v)
        else:
            kwargs[k] = v
    if "task" in kwargs and isinstance(kwargs["task"], TaskSpec):
        kwargs["task"] = TaskSpec(
            **{**asdict(kwargs["task"]), "weights": tuple(kwargs["task"].weights)}
        )
    return RunConfig(**kwargs)


def environment_fingerprint() -> dict[str, Any]:
    """Everything needed to attribute a number to the machine that produced it."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - provenance must never break a run
        commit, dirty = "unknown", False

    versions: dict[str, str] = {}
    for mod in ("numpy", "torch", "clingo", "scipy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "absent"

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
    }
