"""Concept spaces: the latent discrete vocabulary a neuro-symbolic model must ground."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptSpace:
    """The latent concept vocabulary and arity of a neuro-symbolic task.

    Attributes:
        k: size of the concept vocabulary, i.e. concepts are drawn from ``range(k)``.
        n_slots: how many concepts each input carries (2 for MNIST-Addition pairs).
        shared_encoder: whether one encoder is applied to every slot. This is the
            standard setting and it is what makes a *single* relabelling
            ``alpha: [k] -> [k]`` the right object to reason about. With a
            per-slot encoder the shortcut object is an ``n_slots``-tuple of maps
            instead, which enlarges the search space enormously -- see
            ``oracle.base.RSMode``.
    """

    k: int
    n_slots: int
    shared_encoder: bool = True

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots}")

    @property
    def n_tuples(self) -> int:
        """Number of distinct concept tuples, ``k ** n_slots``."""
        return self.k**self.n_slots

    @property
    def n_shared_maps(self) -> int:
        """Size of the search space of shared relabellings, ``k ** k``."""
        return self.k**self.k
