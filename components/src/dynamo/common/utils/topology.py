# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology domain utilities for topology-aware KV transfer routing.

Workers read their topology placement (e.g. zone, rack) from a file
projected by the Kubernetes Downward API volume, and read KV transfer
policy from environment variables. Both are published via MDC so the
router can enforce topology constraints per worker set.

Environment variables (set by the operator):
    DYN_TOPOLOGY_ENABLED: Set to "true" to enable topology reading.
    DYN_TOPOLOGY_MOUNT_PATH: Directory where the Downward API volume is
        mounted (default: /etc/dynamo/topology).
    DYN_TOPOLOGY_DOMAIN: The topology domain name to read (e.g. "zone").
        The file at {mount_path}/{domain} contains the topology value.
    DYN_KV_TRANSFER_DOMAIN: Which topology domain the router should enforce
        for KV transfer constraints (e.g. "zone").
    DYN_KV_TRANSFER_NO_MATCH_POLICY: Behavior when no same-domain decode
        workers exist ("fail" or "fallback", default: "fail").
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_TOPOLOGY_ENABLED_VAR = "DYN_TOPOLOGY_ENABLED"
_TOPOLOGY_MOUNT_PATH_VAR = "DYN_TOPOLOGY_MOUNT_PATH"
_TOPOLOGY_DOMAIN_VAR = "DYN_TOPOLOGY_DOMAIN"
_KV_TRANSFER_DOMAIN_VAR = "DYN_KV_TRANSFER_DOMAIN"
_KV_TRANSFER_NO_MATCH_POLICY_VAR = "DYN_KV_TRANSFER_NO_MATCH_POLICY"
_DEFAULT_MOUNT_PATH = "/etc/dynamo/topology"
_POLL_INTERVAL_SECS = 1.0
_POLL_TIMEOUT_SECS = 30.0

logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """Topology and KV transfer policy configuration read at worker startup."""

    topology_domains: dict[str, str] = field(default_factory=dict)
    kv_transfer_domain: str | None = None
    kv_transfer_no_match_policy: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.topology_domains)


def _read_topology_file(topology_file: Path) -> str | None:
    """Try to read a non-empty value from the topology file.

    Returns the stripped value if the file exists and is non-empty,
    otherwise returns None.
    """
    try:
        value = topology_file.read_text().strip()
        return value if value else None
    except FileNotFoundError:
        return None


def read_topology_config(
    poll_interval: float = _POLL_INTERVAL_SECS,
    poll_timeout: float = _POLL_TIMEOUT_SECS,
) -> TopologyConfig:
    """Read topology config from Downward API volume and env vars.

    The operator injects env vars for topology location and transfer policy:
      - DYN_TOPOLOGY_ENABLED=true
      - DYN_TOPOLOGY_MOUNT_PATH=/etc/dynamo/topology
      - DYN_TOPOLOGY_DOMAIN=zone
      - DYN_KV_TRANSFER_DOMAIN=zone
      - DYN_KV_TRANSFER_NO_MATCH_POLICY=fail

    The topology value is read from the file at {mount_path}/{domain}.
    Since the Downward API volume reflects live label updates, this
    function polls until the file has content, sleeping between retries.

    Args:
        poll_interval: Seconds between retries (default: 1.0).
        poll_timeout: Maximum seconds to wait (default: 30.0).

    Returns:
        TopologyConfig with topology domains and transfer policy.
        Empty config if topology is not enabled.

    Raises:
        SystemExit: If DYN_TOPOLOGY_ENABLED=true but the topology file is
            still missing or empty after the timeout, or the domain env var
            is not set.
    """
    enabled = os.environ.get(_TOPOLOGY_ENABLED_VAR, "").lower()
    if enabled != "true":
        return TopologyConfig()

    domain = os.environ.get(_TOPOLOGY_DOMAIN_VAR, "").strip().lower()
    if not domain:
        logger.error(
            "DYN_TOPOLOGY_ENABLED=true but %s is not set. "
            "The operator must set this to the topology domain name "
            "(e.g. 'zone'). Exiting.",
            _TOPOLOGY_DOMAIN_VAR,
        )
        sys.exit(1)

    mount_path = os.environ.get(_TOPOLOGY_MOUNT_PATH_VAR, _DEFAULT_MOUNT_PATH)
    topology_file = Path(mount_path) / domain

    # Poll until the file has content or timeout expires.
    deadline = time.monotonic() + poll_timeout
    value = _read_topology_file(topology_file)
    while value is None and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        logger.info(
            "Waiting for topology file %s (%.0fs remaining)...",
            topology_file,
            remaining,
        )
        time.sleep(min(poll_interval, max(remaining, 0)))
        value = _read_topology_file(topology_file)

    if value is None:
        logger.error(
            "DYN_TOPOLOGY_ENABLED=true but topology file %s was not populated "
            "within %.0fs. This indicates the pod label was never projected "
            "via the Downward API. Exiting.",
            topology_file,
            poll_timeout,
        )
        sys.exit(1)

    # Read transfer policy from env vars.
    kv_transfer_domain = os.environ.get(_KV_TRANSFER_DOMAIN_VAR, "").strip().lower() or None
    kv_transfer_no_match_policy = (
        os.environ.get(_KV_TRANSFER_NO_MATCH_POLICY_VAR, "").strip().lower() or None
    )

    config = TopologyConfig(
        topology_domains={domain: value},
        kv_transfer_domain=kv_transfer_domain,
        kv_transfer_no_match_policy=kv_transfer_no_match_policy,
    )
    logger.info("Topology config: %s (from %s)", config, topology_file)
    return config
