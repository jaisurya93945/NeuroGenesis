"""Run execution: config in, one append-only JSONL record out.

Every record carries enough provenance to attribute the number to the exact code,
data and machine that produced it: commit sha, config hash, seeds, package
versions, hardware, runtime. Re-running a config hash already present in the store
is a no-op, so a multi-hour sweep resumes safely after any interruption.

Raw records are never edited by hand. Tables and figures are derived from them by
``scripts/``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .config import RunConfig, environment_fingerprint
from .data import mnist
from .data.tuples import make_synthetic_codebook, render_mnist, render_synthetic
from .generators.algebraic import addition_task, modular_task
from .models.encoders import build_encoder
from .oracle import enumerate as en
from .oracle.base import RSResult
from .tasks import Task
from .train.loop import TrainConfig, train

DEFAULT_STORE = Path("results/runs/runs.jsonl")

ORACLE_ARGS = dict(mode="shared", closure="total", allow_noninjective=True)


def build_task(cfg: RunConfig) -> Task:
    """Materialise the task named by the config."""
    spec = cfg.task
    if spec.family == "algebraic":
        task = modular_task(list(spec.weights), spec.m, k=spec.k)
    elif spec.family == "addition":
        k = spec.k or spec.m
        task = addition_task(k=k, n_slots=len(spec.weights))
    else:
        raise ValueError(f"unknown task family {spec.family!r}")

    if spec.support_density < 1.0:
        rng = np.random.default_rng(spec.seed)
        grid = task.all_tuples()
        n_keep = max(1, int(round(spec.support_density * len(grid))))
        keep = grid[rng.choice(len(grid), size=n_keep, replace=False)]
        task = Task(
            name=f"{task.name}_d{spec.support_density}_s{spec.seed}",
            space=task.space,
            label_table=task.label_table,
            support=keep,
            n_labels=task.n_labels,
            meta={**task.meta, "support_density": spec.support_density, "task_seed": spec.seed},
        )
    return task


def build_datasets(cfg: RunConfig, task: Task):
    """Render train/val/test tuple datasets for the configured perception tier."""
    rng = np.random.default_rng(cfg.data.data_seed)
    if cfg.data.tier == "M":
        data = mnist.load()
        return (
            render_mnist(task, cfg.data.n_train, "train", data, rng),
            {
                "val": render_mnist(task, cfg.data.n_val, "val", data, rng),
                "test": render_mnist(task, cfg.data.n_test, "test", data, rng),
            },
        )
    if cfg.data.tier == "S":
        book = make_synthetic_codebook(task.space.k, cfg.data.dim, np.random.default_rng(12345))
        mk = lambda n, s: render_synthetic(task, n, book, cfg.data.noise, rng, s)  # noqa: E731
        return mk(cfg.data.n_train, "train"), {
            "val": mk(cfg.data.n_val, "val"),
            "test": mk(cfg.data.n_test, "test"),
        }
    raise ValueError(f"unknown data tier {cfg.data.tier!r}")


def existing_hashes(store: Path = DEFAULT_STORE) -> set[str]:
    """Config hashes already present, so a sweep can resume."""
    if not Path(store).exists():
        return set()
    out = set()
    with open(store) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["config_hash"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


def load_runs(store: Path = DEFAULT_STORE) -> list[dict]:
    """Read every run record."""
    if not Path(store).exists():
        return []
    with open(store) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(cfg: RunConfig, store: Path | None = DEFAULT_STORE, verbose: bool = False) -> dict:
    """Execute one run and append its record. Returns the record."""
    t0 = time.perf_counter()
    task = build_task(cfg)
    rs: RSResult = en.rs_set(task, **ORACLE_ARGS)
    train_ds, eval_sets = build_datasets(cfg, task)

    in_dim = cfg.data.dim if cfg.data.tier == "S" else None
    enc = build_encoder(cfg.model.encoder, k=task.space.k, **({"in_dim": in_dim} if in_dim else {}))
    tcfg = TrainConfig(
        epochs=cfg.optim.epochs,
        batch_size=cfg.optim.batch_size,
        lr=cfg.optim.lr,
        lr_final=cfg.optim.lr_final,
        weight_decay=cfg.optim.weight_decay,
        encoder=cfg.model.encoder,
        device=cfg.device,
        num_threads=cfg.num_threads,
        log_every=5 if verbose else 0,
    )
    res = train(task, enc, train_ds, eval_sets, tcfg, cfg.model.init_seed, rs=rs)

    record = {
        "config_hash": cfg.config_hash(),
        "experiment": cfg.experiment,
        "config": cfg.to_dict(),
        "task": {
            "name": task.name,
            "content_hash": task.content_hash(),
            "k": task.space.k,
            "n_slots": task.space.n_slots,
            "n_labels": task.n_labels,
            "support_size": int(len(task.support)),
            "support_density": task.support_density,
            "label_entropy": task.label_entropy(),
            "meta": {k: v for k, v in task.meta.items() if k != "analytic_rs_count"},
            "analytic_rs_count": task.meta.get("analytic_rs_count"),
        },
        "oracle": {
            "rs_count": rs.count,
            "is_identifiable": rs.is_identifiable,
            "n_permutations": rs.n_permutations,
            "n_collapsing": rs.n_collapsing,
            "truncated": rs.truncated,
            "elapsed_s": rs.elapsed_s,
        },
        "metrics": {name: m.to_dict() for name, m in res.metrics.items()},
        "history": res.history,
        "runtime_s": res.runtime_s,
        "total_s": time.perf_counter() - t0,
        "n_params": res.n_params,
        "env": environment_fingerprint(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if store is not None:
        store = Path(store)
        store.parent.mkdir(parents=True, exist_ok=True)
        with open(store, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return record


def run_many(
    configs: Iterable[RunConfig],
    store: Path | None = DEFAULT_STORE,
    verbose: bool = True,
) -> Iterator[dict]:
    """Run a sweep, skipping configs already present in the store."""
    done = existing_hashes(store) if store else set()
    configs = list(configs)
    for i, cfg in enumerate(configs, 1):
        h = cfg.config_hash()
        if h in done:
            if verbose:
                print(f"[{i}/{len(configs)}] skip {h} (already run)")
            continue
        if verbose:
            print(
                f"[{i}/{len(configs)}] run  {h} {cfg.task.family} w={cfg.task.weights} "
                f"seed={cfg.model.init_seed}",
                flush=True,
            )
        rec = run(cfg, store=store)
        if verbose:
            m = rec["metrics"]["test"]
            print(
                f"    |RS|={rec['oracle']['rs_count']:3d}  Acc(Y)={m['acc_y']:.3f}  "
                f"Acc(C)={m['acc_c']:.3f}  in_RS={m['rs_membership']}  "
                f"{rec['runtime_s']:.0f}s",
                flush=True,
            )
        yield rec
