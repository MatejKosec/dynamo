// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::protocols::tensor;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DisaggregatedEndpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRuntimeConfig {
    pub total_kv_blocks: Option<u64>,

    pub max_num_seqs: Option<u64>,

    pub max_num_batched_tokens: Option<u64>,

    pub tool_call_parser: Option<String>,

    pub reasoning_parser: Option<String>,

    /// When true, strip tool definitions from the chat template when tool_choice is "none".
    #[serde(default = "default_exclude_tools_when_tool_choice_none")]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Starting rank of data parallel ranks for this worker (0 if DP not enabled)
    #[serde(default = "default_data_parallel_start_rank")]
    pub data_parallel_start_rank: u32,

    /// Total number of data parallel ranks for this worker (1 if DP not enabled)
    #[serde(default = "default_data_parallel_size")]
    pub data_parallel_size: u32,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state (default: true)
    #[serde(default = "default_local_indexer")]
    pub enable_local_indexer: bool,

    /// Mapping of engine-specific runtime configs
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub runtime_data: HashMap<String, serde_json::Value>,

    // Provide tensor model config in the case where the model type is Tensor.
    // Currently use JSON object for convinence, the programmatic way is to
    // define the model config struct as part of the tensor protocol and
    // import it here.
    // [gluo TODO] switch to ModelConfig if desired and workout a way to
    // prepare it in a convinent way, the protobuf library used by tonic
    // doesn't provide JSON parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_model_config: Option<tensor::TensorModelConfig>,

    /// Bootstrap endpoint for disaggregated serving (prefill workers publish this)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_endpoint: Option<DisaggregatedEndpoint>,

    #[serde(default = "default_eagle")]
    pub enable_eagle: bool,

    /// Topology domain labels for this worker (e.g. {"zone": "us-east-1a", "rack": "rack1"}).
    /// Populated from DYN_TOPOLOGY_* environment variables at worker startup.
    /// Used by topology-aware routing to constrain KV cache transfers within a topology domain.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub topology_domains: HashMap<String, String>,

    /// Topology domain to enforce for KV-cache transfers (e.g. "zone").
    /// Set by the operator via DYN_TOPOLOGY_DOMAIN env var. When present, the router
    /// constrains decode worker selection to workers sharing the same domain value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_domain: Option<String>,

    /// Policy when no decode workers share the prefill worker's topology domain.
    /// "fail" (default): return an error. "fallback": allow cross-domain transfer.
    /// Set by the operator via DYN_ROUTER_KV_TRANSFER_NO_MATCH_POLICY env var.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_no_match_policy: Option<String>,
}

const fn default_data_parallel_start_rank() -> u32 {
    0
}

const fn default_data_parallel_size() -> u32 {
    1
}

const fn default_local_indexer() -> bool {
    true
}

const fn default_exclude_tools_when_tool_choice_none() -> bool {
    true
}

const fn default_eagle() -> bool {
    false
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            total_kv_blocks: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: default_exclude_tools_when_tool_choice_none(),
            data_parallel_start_rank: default_data_parallel_start_rank(),
            data_parallel_size: default_data_parallel_size(),
            enable_local_indexer: true,
            runtime_data: HashMap::new(),
            tensor_model_config: None,
            disaggregated_endpoint: None,
            enable_eagle: false,
            topology_domains: HashMap::new(),
            kv_transfer_domain: None,
            kv_transfer_no_match_policy: None,
        }
    }
}

impl dynamo_kv_router::WorkerConfigLike for ModelRuntimeConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn topology_domains(&self) -> Option<&HashMap<String, String>> {
        if self.topology_domains.is_empty() {
            None
        } else {
            Some(&self.topology_domains)
        }
    }

    fn kv_transfer_domain(&self) -> Option<&str> {
        self.kv_transfer_domain.as_deref()
    }

    fn kv_transfer_no_match_policy(&self) -> Option<&str> {
        self.kv_transfer_no_match_policy.as_deref()
    }
}

impl ModelRuntimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_engine_specific<T: Serialize>(&mut self, key: &str, value: T) -> anyhow::Result<()> {
        self.runtime_data
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    pub fn get_engine_specific<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        if let Some(value) = self.runtime_data.get(key) {
            Ok(Some(serde_json::from_value(value.clone())?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::WorkerConfigLike;

    #[test]
    fn test_serde_round_trip_with_topology_domains() {
        let mut config = ModelRuntimeConfig::default();
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());
        config
            .topology_domains
            .insert("rack".to_string(), "rack1".to_string());

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModelRuntimeConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.topology_domains.len(), 2);
        assert_eq!(deserialized.topology_domains["zone"], "us-east-1a");
        assert_eq!(deserialized.topology_domains["rack"], "rack1");
    }

    #[test]
    fn test_serde_empty_topology_domains_omitted() {
        let config = ModelRuntimeConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();

        // Empty topology_domains should not appear in serialized output
        assert!(
            !serialized.contains("topology_domains"),
            "empty topology_domains should be skipped during serialization, got: {serialized}"
        );
    }

    #[test]
    fn test_serde_backward_compat_deserialize_without_topology_domains() {
        // Simulate a config serialized before topology_domains existed
        let json = r#"{
            "total_kv_blocks": 100,
            "max_num_seqs": 32,
            "max_num_batched_tokens": null,
            "tool_call_parser": null,
            "reasoning_parser": null
        }"#;

        let config: ModelRuntimeConfig = serde_json::from_str(json).unwrap();
        assert!(config.topology_domains.is_empty());
    }

    #[test]
    fn test_worker_config_like_topology_domains_with_data() {
        let mut config = ModelRuntimeConfig::default();
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());

        let domains = config.topology_domains();
        assert!(domains.is_some());
        let domains = domains.unwrap();
        assert_eq!(domains.len(), 1);
        assert_eq!(domains["zone"], "us-east-1a");
    }

    #[test]
    fn test_worker_config_like_topology_domains_empty_returns_none() {
        let config = ModelRuntimeConfig::default();
        assert!(config.topology_domains.is_empty());
        assert!(config.topology_domains().is_none());
    }

    #[test]
    fn test_serde_round_trip_preserves_all_fields() {
        let mut config = ModelRuntimeConfig {
            total_kv_blocks: Some(500),
            max_num_seqs: Some(64),
            max_num_batched_tokens: Some(8192),
            tool_call_parser: Some("hermes".to_string()),
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: false,
            data_parallel_start_rank: 2,
            data_parallel_size: 4,
            enable_local_indexer: false,
            runtime_data: HashMap::new(),
            tensor_model_config: None,
            disaggregated_endpoint: Some(DisaggregatedEndpoint {
                bootstrap_host: Some("10.0.0.1".to_string()),
                bootstrap_port: Some(8080),
            }),
            enable_eagle: true,
            topology_domains: HashMap::new(),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_no_match_policy: Some("fail".to_string()),
        };
        config
            .topology_domains
            .insert("zone".to_string(), "us-west-2b".to_string());

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModelRuntimeConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_serde_kv_transfer_fields_omitted_when_none() {
        let config = ModelRuntimeConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();

        assert!(
            !serialized.contains("kv_transfer_domain"),
            "None kv_transfer_domain should be skipped, got: {serialized}"
        );
        assert!(
            !serialized.contains("kv_transfer_no_match_policy"),
            "None kv_transfer_no_match_policy should be skipped, got: {serialized}"
        );
    }

    #[test]
    fn test_serde_round_trip_with_kv_transfer_policy() {
        let mut config = ModelRuntimeConfig::default();
        config.kv_transfer_domain = Some("zone".to_string());
        config.kv_transfer_no_match_policy = Some("fallback".to_string());

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModelRuntimeConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.kv_transfer_domain.as_deref(), Some("zone"));
        assert_eq!(
            deserialized.kv_transfer_no_match_policy.as_deref(),
            Some("fallback")
        );
    }

    #[test]
    fn test_worker_config_like_kv_transfer_domain() {
        let mut config = ModelRuntimeConfig::default();
        assert!(config.kv_transfer_domain().is_none());

        config.kv_transfer_domain = Some("zone".to_string());
        assert_eq!(config.kv_transfer_domain(), Some("zone"));
    }

    #[test]
    fn test_worker_config_like_kv_transfer_no_match_policy() {
        let mut config = ModelRuntimeConfig::default();
        assert!(config.kv_transfer_no_match_policy().is_none());

        config.kv_transfer_no_match_policy = Some("fail".to_string());
        assert_eq!(config.kv_transfer_no_match_policy(), Some("fail"));
    }

    #[test]
    fn test_serde_backward_compat_without_kv_transfer_fields() {
        let json = r#"{
            "total_kv_blocks": 100,
            "max_num_seqs": 32
        }"#;

        let config: ModelRuntimeConfig = serde_json::from_str(json).unwrap();
        assert!(config.kv_transfer_domain.is_none());
        assert!(config.kv_transfer_no_match_policy.is_none());
    }
}
