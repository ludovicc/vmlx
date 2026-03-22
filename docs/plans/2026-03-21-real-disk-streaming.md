# Real Disk Streaming — Layer-by-Layer SSD Paging

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make models exceeding physical RAM runnable via per-layer synchronous GPU execution + reduced wired memory limit, enabling macOS SSD paging.

**Architecture:** (1) Wrap each transformer layer with sync barrier after execution, (2) Lower wired memory limit so macOS can page idle layer weights to SSD. Metal executes one layer at a time while macOS manages SSD paging at ~7.4GB/s.

---

## Why Current Approach Fails

MLX builds entire forward pass as single lazy graph. Metal needs ALL layer weights resident simultaneously. Raising memory limit allows virtual allocation but Metal command buffer still references everything at once = OOM.

## The Fix: Two Components

### Component 1: StreamingLayerWrapper
Wraps each model.layers[i]. After each layer executes, forces synchronous Metal execution via mlx sync call. This breaks the lazy graph at each layer boundary.

### Component 2: Reduced Wired Limit  
Lower mx.set_wired_limit from max_recommended_working_set_size to ~3 layers + KV cache. Tells macOS that model weights beyond wired limit CAN be paged to SSD.

---

## Files to Create/Modify

### Create: vmlx_engine/utils/streaming_wrapper.py
- StreamingLayerWrapper class (wraps layer, forces sync after execution)
- apply_streaming_layers(model) — finds model.layers and wraps each one
- compute_streaming_wired_limit(model, n_layers=3) — calculates proper wired limit

### Modify: vmlx_engine/server.py
- In load_model(), after model loads: apply wrapper + set wired limit when stream_from_disk=True

### Modify: vmlx_engine/mllm_batch_generator.py  
- Skip wired limit override (set_wired_limit to max) when _is_streaming=True
- Currently only skips cache limit — must also skip wired limit

### Modify: tests/test_disk_streaming.py
- Test wrapper creation, layer wrapping, wired limit computation

---

## Model Structure Patterns (all must work)

1. model.model.layers — standard mlx-lm (llama, qwen, mistral)
2. model.language_model.model.layers — VLM wrapper (mistral3, pixtral)
3. model.model.layers with counter-based cache — hybrid SSM (nemotron_h)
4. model.model.pipeline_layers — PipelineMixin (deepseek_v3) — is slice of layers

## Compliance Matrix After Implementation

### Normal mode MUST be unaffected:
- No wrappers applied
- Wired limit at max (default)
- All caching works
- Full speed inference

### Stream mode checks:
- All layers wrapped
- Wired limit = ~3 layers (not max)
- Model exceeding RAM generates tokens
- Memory stays bounded
- Sleep/wake re-applies wrapper
- All model types work (LLM, MoE, hybrid SSM, VLM, JANG)
- All API endpoints work (just slower)

### Edge cases:
- Model fits in RAM: works but slower
- Model 20% over RAM: works, ~2-5 tok/s
- Model 100% over RAM: works, ~0.5-1 tok/s
- Long prefill: slow but works
- System memory pressure: macOS handles
