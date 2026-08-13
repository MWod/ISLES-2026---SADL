"""Network subpackage - custom architectures for the ISLES 2026 ensemble.

The cohort-conditioned mixture-of-experts subclass is loaded lazily from
:mod:`nnunet_isles.networks.plainconv_with_cohort_moe` inside
IslesTrainerCohortMoE.build_network_architecture, so no eager import
happens here.
"""

__all__ = []
