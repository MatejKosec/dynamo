/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestTopologyLabelReconciler_CopiesToPod(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1a"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: map[string]string{},
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	var patched corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &patched))
	assert.Equal(t, "us-east-1a", patched.Labels["topology.kubernetes.io/zone"])
}

func TestTopologyLabelReconciler_SkipsIfLabelExists(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"topology.kubernetes.io/zone": "us-east-1a"},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: map[string]string{
				"topology.kubernetes.io/zone": "us-east-1a",
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestTopologyLabelReconciler_SkipsUnscheduledPod(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
		},
		Spec: corev1.PodSpec{}, // No NodeName yet
	}

	cl := fake.NewClientBuilder().WithObjects(pod).Build()
	r := &TopologyLabelReconciler{Client: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestTopologyLabelReconciler_SkipsIfNodeMissingLabel(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{}, // Missing the topology label
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationTopologyLabelKey: "topology.kubernetes.io/zone",
			},
			Labels: map[string]string{},
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(node, pod).Build()
	r := &TopologyLabelReconciler{Client: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	// Pod should NOT have the label
	var patched corev1.Pod
	require.NoError(t, cl.Get(context.Background(), types.NamespacedName{Name: "worker-abc", Namespace: "default"}, &patched))
	assert.NotContains(t, patched.Labels, "topology.kubernetes.io/zone")
}

func TestTopologyLabelReconciler_SkipsPodWithoutAnnotation(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-abc",
			Namespace: "default",
			// No topology-label-key annotation
		},
		Spec: corev1.PodSpec{NodeName: "node-1"},
	}

	cl := fake.NewClientBuilder().WithObjects(pod).Build()
	r := &TopologyLabelReconciler{Client: cl}

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "worker-abc", Namespace: "default"},
	})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}
