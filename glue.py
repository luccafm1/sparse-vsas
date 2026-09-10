from __future__ import annotations

"""Native partial and total Glue"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sj_federated_networks_operator_v9 import (
    _graph_snapshot,
    OperatorGlueResult,
    fit_operator_sj_glue,
)


_CONFLICT_POLICIES = {"require_equal", "prefer_source", "prefer_target"}


def _core_signature(core: Any) -> tuple:
    signature = getattr(core, "grammar_signature", None)
    if signature is None:
        raise TypeError("core does not expose an SJ grammar_signature")
    return tuple(signature)


def _validate_codes(codes: torch.Tensor, groups: int, card: int, name: str) -> None:
    if not torch.is_tensor(codes):
        raise TypeError(f"{name} must be a torch.Tensor")
    if codes.ndim != 2 or codes.shape[1] != groups:
        raise ValueError(f"{name} must have shape [batch, {groups}]")
    if torch.is_floating_point(codes) or codes.dtype == torch.bool:
        raise TypeError(f"{name} must contain integer hard SJ codes")
    if bool(((codes < -1) | (codes >= card)).any()):
        raise ValueError(f"{name} contains a code outside [-1, {card - 1}]")


@dataclass(frozen=True)
class SJTransitionEvidence:
    """``actions[name]`` may be an after-state array sharing rows with ``base``,
    or an explicit ``(before, after)`` pair.  
    
    Trivially, the latter supports trajectory datasets whose pre-state distribution differs by action.
    """

    base: np.ndarray
    actions: Mapping[str, Any]


@dataclass(frozen=True)
class SJGlue:
    """Certified native map from a source algebra into a target algebra.

    ``group_map[i] == j`` maps source group ``i`` to target group ``j``.
    ``value_maps[i][c]`` maps source filler ``c`` into the private filler
    vocabulary of that target group.  ``-1``/``None`` denotes a source-private
    or unresolved atom.  Target groups outside the image remain target-private.
    """

    certificate: OperatorGlueResult
    source_groups: int
    target_groups: int
    cardinality: int
    grammar_signature: tuple
    confirmation_certificate: OperatorGlueResult | None = None

    def __post_init__(self) -> None:
        group_map = self.certificate.group_map
        value_maps = self.certificate.value_maps
        if len(group_map) != self.source_groups or len(value_maps) != self.source_groups:
            raise ValueError("certificate has the wrong number of source groups")
        image = [j for j in group_map if j >= 0]
        if len(image) != len(set(image)):
            raise ValueError("an SJ Glue group map must be injective")
        if any(j >= self.target_groups for j in image):
            raise ValueError("certificate maps outside the target SJ groups")
        for group, values in zip(group_map, value_maps):
            if (group < 0) != (values is None):
                raise ValueError("group and filler maps must have identical support")
            if values is not None and sorted(values) != list(range(self.cardinality)):
                raise ValueError("every resolved filler map must be a permutation")

    @property
    def status(self) -> str:
        if self.confirmation_certificate is not None and not self.is_executable:
            return "independently_unconfirmed"
        return self.certificate.status

    @property
    def is_executable(self) -> bool:
        primary = bool(
            self.certificate.compatible and self.certificate.status == "grounded"
        )
        if self.confirmation_certificate is None:
            return primary
        confirmation = self.confirmation_certificate
        same_map = (
            self.certificate.group_map == confirmation.group_map
            and self.certificate.value_maps == confirmation.value_maps
            and self.certificate.source_graph_hash == confirmation.source_graph_hash
            and self.certificate.target_graph_hash == confirmation.target_graph_hash
        )
        return bool(
            primary
            and confirmation.compatible
            and confirmation.status == "grounded"
            and same_map
        )

    @property
    def resolved_source_groups(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, (j, values) in enumerate(
                zip(self.certificate.group_map, self.certificate.value_maps)
            )
            if j >= 0 and values is not None
        )

    @property
    def resolved_target_groups(self) -> tuple[int, ...]:
        return tuple(self.certificate.group_map[i] for i in self.resolved_source_groups)

    @property
    def source_coverage(self) -> float:
        return len(self.resolved_source_groups) / max(self.source_groups, 1)

    @property
    def target_coverage(self) -> float:
        return len(self.resolved_target_groups) / max(self.target_groups, 1)

    @property
    def kind(self) -> str:
        if (
            len(self.resolved_source_groups) == self.source_groups
            and len(self.resolved_target_groups) == self.target_groups
        ):
            return "total"
        return "partial"

    def _require_executable(self, allow_candidate: bool) -> None:
        if not self.is_executable and not allow_candidate:
            raise RuntimeError(
                f"refusing non-grounded SJ Glue ({self.status}): "
                f"{self.certificate.reason}"
            )

    def translate_codes(
        self,
        source_codes: torch.Tensor,
        *,
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        """Translate supported hard atoms & emit ``-1`` everywhere else."""
        self._require_executable(allow_candidate)
        _validate_codes(
            source_codes, self.source_groups, self.cardinality, "source_codes"
        )
        translated = torch.full(
            (len(source_codes), self.target_groups),
            -1,
            dtype=source_codes.dtype,
            device=source_codes.device,
        )
        for source_group in self.resolved_source_groups:
            target_group = self.certificate.group_map[source_group]
            values = torch.as_tensor(
                self.certificate.value_maps[source_group],
                dtype=torch.long,
                device=source_codes.device,
            )
            source = source_codes[:, source_group].long()
            active = source >= 0
            translated[active, target_group] = values[source[active]].to(
                translated.dtype
            )
        return translated

    def merge_codes(
        self,
        source_codes: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        conflict: str = "require_equal",
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        """Compose imported atoms with target-private SJ atoms"""
        if conflict not in _CONFLICT_POLICIES:
            raise ValueError(f"conflict must be one of {sorted(_CONFLICT_POLICIES)}")
        _validate_codes(
            target_codes, self.target_groups, self.cardinality, "target_codes"
        )
        imported = self.translate_codes(source_codes, allow_candidate=allow_candidate)
        if len(imported) != len(target_codes):
            raise ValueError("source_codes and target_codes must have the same batch size")
        merged = target_codes.clone()
        fill = (merged < 0) & (imported >= 0)
        merged[fill] = imported[fill]
        disagree = (merged >= 0) & (imported >= 0) & (merged != imported)
        if bool(disagree.any()):
            if conflict == "require_equal":
                rows, groups = torch.where(disagree)
                sample = list(zip(rows[:4].tolist(), groups[:4].tolist()))
                raise ValueError(f"conflicting shared SJ atoms at {sample}")
            if conflict == "prefer_source":
                merged[disagree] = imported[disagree]
        return merged

    def import_program(
        self,
        target_core: Any,
        source_codes: torch.Tensor,
        *,
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        """Reify translated atoms with the target native VSA algebra"""
        self._validate_target_core(target_core)
        translated = self.translate_codes(source_codes, allow_candidate=allow_candidate)
        return target_core.program_feature_from_codes(translated)

    def compose_program(
        self,
        target_core: Any,
        source_codes: torch.Tensor,
        target_codes: torch.Tensor,
        *,
        conflict: str = "require_equal",
        allow_candidate: bool = False,
    ) -> torch.Tensor:
        """Merge shared/private hard atoms and ask the target to bind them"""
        self._validate_target_core(target_core)
        merged = self.merge_codes(
            source_codes,
            target_codes,
            conflict=conflict,
            allow_candidate=allow_candidate,
        )
        return target_core.program_feature_from_codes(merged)

    def _validate_target_core(self, target_core: Any) -> None:
        if int(target_core.G) != self.target_groups or int(target_core.C) != self.cardinality:
            raise ValueError("target core shape does not match the SJ Glue certificate")
        if _core_signature(target_core) != self.grammar_signature:
            raise ValueError("target core uses a different SJ program grammar")
        fingerprint = self.certificate.target_graph_hash
        if fingerprint is not None and _graph_snapshot(target_core, True).graph_hash != fingerprint:
            raise ValueError("target executing graph changed after Glue certification")

    def summary(self) -> dict[str, Any]:
        record = asdict(self.certificate)
        record.update(
            {
                "status": self.status,
                "compatible": self.is_executable,
                "kind": self.kind,
                "source_groups": self.source_groups,
                "target_groups": self.target_groups,
                "cardinality": self.cardinality,
                "source_coverage": self.source_coverage,
                "target_coverage": self.target_coverage,
                "grammar_signature": list(self.grammar_signature),
                "is_executable": self.is_executable,
                "independent_confirmation": (
                    asdict(self.confirmation_certificate)
                    if self.confirmation_certificate is not None
                    else None
                ),
                "independent_map_agreement": (
                    self.certificate.group_map
                    == self.confirmation_certificate.group_map
                    and self.certificate.value_maps
                    == self.confirmation_certificate.value_maps
                    if self.confirmation_certificate is not None
                    else None
                ),
            }
        )
        return record


def fit_sj_glue(
    base_a: np.ndarray,
    actions_a: Mapping[str, np.ndarray],
    core_a: Any,
    base_b: np.ndarray,
    actions_b: Mapping[str, np.ndarray],
    core_b: Any,
    *,
    train_actions: Sequence[str],
    validation_actions: Sequence[str] = (),
    **certificate_options: Any,
) -> SJGlue:
    """Fit and certify a native SJ morphism from A into B.

    Rows are never paired between A and B.  ``base_a`` only pairs with each
    array in ``actions_a``; equivalently for B.  Train and validation action
    names are public intervention identities, not observation correspondences.
    """
    signature_a = _core_signature(core_a)
    signature_b = _core_signature(core_b)
    result = fit_operator_sj_glue(
        base_a,
        actions_a,
        core_a,
        base_b,
        actions_b,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **certificate_options,
    )
    signature = signature_a if signature_a == signature_b else signature_a
    return SJGlue(
        certificate=result,
        source_groups=int(core_a.G),
        target_groups=int(core_b.G),
        cardinality=int(core_a.C),
        grammar_signature=signature,
    )


def certify_sj_glue(
    fit_a: SJTransitionEvidence,
    fit_b: SJTransitionEvidence,
    confirmation_a: SJTransitionEvidence,
    confirmation_b: SJTransitionEvidence,
    core_a: Any,
    core_b: Any,
    *,
    train_actions: Sequence[str],
    validation_actions: Sequence[str] = (),
    **certificate_options: Any,
) -> SJGlue:
    """Require the complete morphism to replicate on an independent corpus"""
    fit_options = dict(certificate_options)
    confirmation_options = dict(certificate_options)
    if "seed" in confirmation_options:
        confirmation_options["seed"] = int(confirmation_options["seed"]) + 100_003
    primary = fit_sj_glue(
        fit_a.base,
        fit_a.actions,
        core_a,
        fit_b.base,
        fit_b.actions,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **fit_options,
    )
    confirmation = fit_sj_glue(
        confirmation_a.base,
        confirmation_a.actions,
        core_a,
        confirmation_b.base,
        confirmation_b.actions,
        core_b,
        train_actions=train_actions,
        validation_actions=validation_actions,
        **confirmation_options,
    )
    return SJGlue(
        certificate=primary.certificate,
        source_groups=primary.source_groups,
        target_groups=primary.target_groups,
        cardinality=primary.cardinality,
        grammar_signature=primary.grammar_signature,
        confirmation_certificate=confirmation.certificate,
    )
