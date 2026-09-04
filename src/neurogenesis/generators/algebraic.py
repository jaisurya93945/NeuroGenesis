"""Modular-linear task family: ``y = (sum_i w_i * c_i) mod m``.

This family is the backbone of the project because its reasoning-shortcut set has
a **closed form**, which buys two things at once: free known-answer tests for the
oracle, and an experimental family in which ``|RS|`` can be varied while every
plausible confound is held exactly constant.

Derivation (see ``RESEARCH.md`` for the full statement). Write ``d(x) = alpha(x) - x``
mod ``m``. Preservation of the label on the full grid means, for all ``c``::

    sum_i w_i * d(c_i) = 0   (mod m)

Varying one coordinate with the others fixed forces ``w_i * d(.)`` to be constant;
when ``gcd(w_i, m) = 1`` that means ``d`` itself is a constant ``u``. Substituting
back gives ``u * sum_i w_i = 0 (mod m)``, which has exactly ``gcd(sum_i w_i, m)``
solutions in ``Z_m``. Hence::

    |RS| = gcd(sum_i w_i, m)        (full support, every w_i coprime to m)

and every shortcut is the cyclic shift ``alpha(x) = x + u mod m``. Because a
non-zero shift maps *every* concept to a wrong one, a model that lands on a
shortcut has concept accuracy exactly 0 -- the outcome is bimodal rather than
noisy, which is what makes the design statistically sharp.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..tasks import Task, task_from_fn


def analytic_rs_count(weights: Sequence[int], m: int) -> int | None:
    """Closed-form ``|RS|`` for the modular-linear family on full support.

    Returns ``None`` when the precondition (every weight coprime to ``m``) fails,
    because then the derivation above does not apply and the oracle must be used.
    """
    if any(math.gcd(w % m, m) != 1 for w in weights):
        return None
    return math.gcd(sum(weights) % m, m)


def modular_task(
    weights: Sequence[int],
    m: int,
    *,
    k: int | None = None,
    name: str | None = None,
    support: np.ndarray | None = None,
) -> Task:
    """Build ``y = (sum_i w_i c_i) mod m`` over a concept vocabulary of size ``k``.

    ``k`` defaults to ``m`` (concepts and labels share an alphabet), which is the
    setting the closed form covers.
    """
    weights = list(weights)
    k = m if k is None else k
    n = len(weights)
    w = np.array(weights, dtype=np.int64)

    def fn(c: tuple[int, ...]) -> int:
        return int((w * np.array(c, dtype=np.int64)).sum() % m)

    predicted = analytic_rs_count(weights, m) if (k == m and support is None) else None
    return task_from_fn(
        name=name or f"mod{m}_w{'_'.join(map(str, weights))}",
        k=k,
        n_slots=n,
        fn=fn,
        n_labels=m,
        support=support,
        meta={
            "family": "algebraic",
            "weights": weights,
            "m": m,
            "analytic_rs_count": predicted,
        },
    )


def addition_task(k: int = 10, n_slots: int = 2, *, support: np.ndarray | None = None) -> Task:
    """Plain integer addition ``y = sum_i c_i`` -- the classic MNIST-Addition knowledge.

    Unlike the modular family this is not wrap-around, so on full support the
    ``c_1 = c_2`` diagonal forces ``2*alpha(a) = 2*a`` over the integers and the
    task is identifiable. Restricting the support is what makes it interesting.
    """
    return task_from_fn(
        name=f"addition_k{k}_n{n_slots}",
        k=k,
        n_slots=n_slots,
        fn=lambda c: int(sum(c)),
        n_labels=n_slots * (k - 1) + 1,
        support=support,
        meta={"family": "addition", "analytic_rs_count": 1 if support is None else None},
    )


def mod10_family() -> list[Task]:
    """The headline E1 family: ``y = (c1 + w*c2) mod 10`` for ``w = 1..9``.

    Across the family, support, ``|Y|``, the label marginal, ``I(C;Y)`` and
    perceptual difficulty are *identical*; only ``|RS|`` moves, over
    ``{1, 2, 5, 10}``. The confound is controlled by construction rather than by
    statistical adjustment, which is the design's main methodological claim.
    """
    return [modular_task([1, w], 10, name=f"mod10_w{w}") for w in range(1, 10)]
