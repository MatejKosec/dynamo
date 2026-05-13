# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for topology domain utilities.

Tests the read_topology_config() function that reads topology from a
Downward API volume file and KV transfer policy from env vars at worker
startup for topology-aware KV transfer routing.

These tests import topology.py directly (bypassing the dynamo package hierarchy)
so they work without GPU, CUDA, or any backend installed.
"""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# ---------------------------------------------------------------------------
# Module loading: import topology without triggering the full dynamo package
# (which requires dynamo.llm, CUDA, etc.)
# ---------------------------------------------------------------------------
_TOPOLOGY_PY = Path(__file__).resolve().parents[2] / "utils" / "topology.py"


def _load_topology_module():
    """Load topology.py as a standalone module."""
    spec = importlib.util.spec_from_file_location("topology", _TOPOLOGY_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


topology = _load_topology_module()
read_topology_config = topology.read_topology_config


class TestReadTopologyConfig:
    """Tests for read_topology_config()."""

    def test_returns_empty_when_not_enabled(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED is not set, returns empty config."""
        monkeypatch.delenv("DYN_TOPOLOGY_ENABLED", raising=False)
        config = read_topology_config()
        assert not config.enabled
        assert config.topology_domains == {}
        assert config.kv_transfer_domain is None
        assert config.kv_transfer_no_match_policy is None

    def test_returns_empty_when_enabled_false(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED=false, returns empty config."""
        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "false")
        config = read_topology_config()
        assert not config.enabled

    def test_returns_empty_when_enabled_empty_string(self, monkeypatch):
        """When DYN_TOPOLOGY_ENABLED='', returns empty config."""
        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "")
        config = read_topology_config()
        assert not config.enabled

    def test_reads_topology_from_file(self, monkeypatch, tmp_path):
        """Reads topology value from Downward API volume file."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
        monkeypatch.delenv("DYN_KV_TRANSFER_DOMAIN", raising=False)
        monkeypatch.delenv("DYN_KV_TRANSFER_NO_MATCH_POLICY", raising=False)

        config = read_topology_config()
        assert config.enabled
        assert config.topology_domains == {"zone": "us-east-1a"}

    def test_reads_transfer_policy_env_vars(self, monkeypatch, tmp_path):
        """Reads KV transfer policy from env vars."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
        monkeypatch.setenv("DYN_KV_TRANSFER_DOMAIN", "zone")
        monkeypatch.setenv("DYN_KV_TRANSFER_NO_MATCH_POLICY", "fallback")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_no_match_policy == "fallback"

    def test_transfer_policy_none_when_env_vars_not_set(self, monkeypatch, tmp_path):
        """Transfer policy fields are None when env vars are not set."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
        monkeypatch.delenv("DYN_KV_TRANSFER_DOMAIN", raising=False)
        monkeypatch.delenv("DYN_KV_TRANSFER_NO_MATCH_POLICY", raising=False)

        config = read_topology_config()
        assert config.kv_transfer_domain is None
        assert config.kv_transfer_no_match_policy is None

    def test_strips_whitespace_from_file_value(self, monkeypatch, tmp_path):
        """Strips whitespace/newlines from the topology file value."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("  us-east-1a\n")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        config = read_topology_config()
        assert config.topology_domains == {"zone": "us-east-1a"}

    def test_domain_key_is_lowercased(self, monkeypatch, tmp_path):
        """Domain key is lowercased even if env var has mixed case."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "ZONE")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        config = read_topology_config()
        assert "zone" in config.topology_domains

    def test_hard_exit_when_domain_env_not_set(self, monkeypatch):
        """Exits when enabled but DYN_TOPOLOGY_DOMAIN is not set."""
        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.delenv("DYN_TOPOLOGY_DOMAIN", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config()
        assert exc_info.value.code == 1

    def test_hard_exit_after_timeout_file_missing(self, monkeypatch, tmp_path):
        """Exits when file never appears within timeout."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config(poll_interval=0.05, poll_timeout=0.15)
        assert exc_info.value.code == 1

    def test_hard_exit_after_timeout_file_empty(self, monkeypatch, tmp_path):
        """Exits when file exists but stays empty within timeout."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        with pytest.raises(SystemExit) as exc_info:
            read_topology_config(poll_interval=0.05, poll_timeout=0.15)
        assert exc_info.value.code == 1

    def test_retry_succeeds_when_file_appears_during_polling(self, monkeypatch, tmp_path):
        """File appears after a delay; retry loop picks it up."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        def write_after_delay():
            time.sleep(0.1)
            (topology_dir / "zone").write_text("us-west-2a")

        writer = threading.Thread(target=write_after_delay)
        writer.start()

        config = read_topology_config(poll_interval=0.05, poll_timeout=2.0)
        writer.join()
        assert config.topology_domains == {"zone": "us-west-2a"}

    def test_retry_succeeds_when_empty_file_gets_content(self, monkeypatch, tmp_path):
        """File exists but is empty; content appears after a delay."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        zone_file = topology_dir / "zone"
        zone_file.write_text("")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))

        def write_after_delay():
            time.sleep(0.1)
            zone_file.write_text("eu-central-1a")

        writer = threading.Thread(target=write_after_delay)
        writer.start()

        config = read_topology_config(poll_interval=0.05, poll_timeout=2.0)
        writer.join()
        assert config.topology_domains == {"zone": "eu-central-1a"}

    def test_transfer_policy_lowercased(self, monkeypatch, tmp_path):
        """Transfer policy env var values are lowercased."""
        topology_dir = tmp_path / "topology"
        topology_dir.mkdir()
        (topology_dir / "zone").write_text("us-east-1a")

        monkeypatch.setenv("DYN_TOPOLOGY_ENABLED", "true")
        monkeypatch.setenv("DYN_TOPOLOGY_DOMAIN", "zone")
        monkeypatch.setenv("DYN_TOPOLOGY_MOUNT_PATH", str(topology_dir))
        monkeypatch.setenv("DYN_KV_TRANSFER_DOMAIN", "ZONE")
        monkeypatch.setenv("DYN_KV_TRANSFER_NO_MATCH_POLICY", "Fallback")

        config = read_topology_config()
        assert config.kv_transfer_domain == "zone"
        assert config.kv_transfer_no_match_policy == "fallback"
