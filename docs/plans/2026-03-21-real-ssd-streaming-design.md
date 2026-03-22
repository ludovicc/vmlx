# Real SSD Streaming — Per-Layer Weight Recycling Design

**Status: IMPLEMENTED — Core weight recycling system complete**

## Problem

Current disk streaming does NOT actually stream from SSD. It uses lazy mmap + raised Metal limits, but Metal still pulls ALL weights into GPU memory on first forward pass. An 18GB model uses 17.5GB of Metal active memory regardless of wired limit settings.

**Proven experimentally:**
- Setting wired limit to 1GB + memory limit to 384GB + cache limit to 0 — Metal still reports 10.29GB active after inference
- The old StreamingLayerWrapper (sync barrier per layer) does NOT cause Metal to release weight buffers
- Wired limit only controls pinning, not allocation

## Solution: Per-Layer Weight Recycling (Proven)

**Core mechanism (experimentally confirmed):**
```
After forward:  Metal active = 17,579MB (all layers loaded)
After freeing:  Metal active =  1,329MB (only embeddings + KV cache)
```

Replacing layer weight arrays with tiny placeholders DOES free Metal GPU memory. The OS/Metal releases buffers when Python drops references + gc.collect().

**The approach:**
1. Load model structure + weights initially
2. Build a layer-to-file mapping (which safetensors file has which layer's params)
3. For JANG models: save transformed weights to temp safetensors (one per layer)
4. Before inference: free all layer weights (keep embeddings + lm_head + KV cache)
5. During decode: for each layer, load weights from file, compute, free weights
6. Metal memory stays bounded to approx 1 layer + embeddings + lm_head + KV cache

## Architecture: Approach B (vmlx_engine level interception)

NOT patching mlx-lm internals. Write a custom generate loop in vmlx_engine that replaces mlx_lm.generate() when SSD streaming is active.

- We own the code — no third-party monkey-patching
- Full control over prefill vs decode behavior
- Can handle all model types (JANG, VLM, hybrid SSM, standard)
- Clean separation: streaming off = standard mlx_lm path, streaming on = our custom path

## JANG Handling: Temp File Save

JANG transforms weights during loading (gate dequant, fc1/fc2 rename, expert restack). On-disk safetensors dont match in-memory weights.

After initial JANG load, save each layers transformed weights to temp safetensors files. Use those for the load/unload cycle. Cleanup on process exit or deep sleep.

## Forward Pass Trace

### SimpleEngine Path (used with disk streaming, max_num_seqs=1):
```
server.py:create_chat_completion()
  -> engine/simple.py:SimpleEngine.chat()
    -> models/llm.py:MLXLanguageModel.generate()
      -> mlx_lm.generate() -> stream_generate() -> generate_step()
        -> PREFILL: model(prompt_chunk, cache=cache) [chunked 2048 at a time]
        -> DECODE:  model(single_token, cache=cache) [1 token per iteration]
          -> Model.__call__():
            h = embed_tokens(inputs)
            for layer, c in zip(self.layers, cache):
              h = layer(h, mask, cache=c)      <-- WEIGHTS USED HERE
            return lm_head(norm(h))
```

### Key locations in mlx_lm/generate.py:
- generate_step() lines 303-466: Main generator (prefill + decode)
- Prefill phase: lines 420-449 (processes prompt in chunks)
- Decode phase: lines 449-466 (one token per loop iteration)
- _model_call() lines 384-390: Actual model(tokens, cache=cache) call

### Key locations in model files (e.g. llama.py):
- Model.__call__: embedding -> inner_model -> lm_head
- InnerModel.__call__: embed_tokens -> layer_loop -> norm
- Layer loop: for layer, c in zip(self.layers, cache): h = layer(h, mask, cache=c)

## Files to Create

### 1. vmlx_engine/utils/ssd_generate.py
Custom generate loop with weight recycling:
- ssd_generate() — replaces mlx_lm.generate() when streaming
- ssd_stream_generate() — streaming variant
- LayerWeightManager — handles load/save/free per-layer weights
- _ssd_prefill() — prefill with all weights loaded
- _ssd_decode_step() — single token with per-layer weight swap

### 2. vmlx_engine/utils/weight_index.py
Weight file mapping:
- build_weight_index(model_path) — maps layer_idx to file+keys
- save_layer_weights(model, layer_idx, output_dir) — saves single layer to safetensors
- load_layer_weights(layer, layer_idx, index) — loads from safetensors into layer
- free_layer_weights(layer) — replaces with zeros, gc.collect()

## Files to Modify

### 3. vmlx_engine/models/llm.py
Add ssd_generate() method that uses custom loop instead of mlx_lm.generate(). Detect _stream_from_disk and route to custom path.

### 4. vmlx_engine/models/mllm.py
Same for VLM models. Vision encoding stays standard, only text generation uses weight recycling.

### 5. vmlx_engine/engine/simple.py
Route to ssd_generate when streaming active.

### 6. vmlx_engine/server.py
Remove old StreamingLayerWrapper application. Replace with new weight manager init. User-controllable settings (no preset rules).

### 7. vmlx_engine/cli.py
Add new CLI args:
- --ssd-memory-budget (MB): how much Metal memory for layer weights
- --ssd-prefetch-layers (int): how many layers to prefetch

### 8. vmlx_engine/utils/streaming_wrapper.py
REPLACE contents entirely. Remove old StreamingLayerWrapper (proven not to work). New: weight recycling manager.

### 9. panel SessionConfigForm.tsx
Replace toggle + slider with granular controls:
- Enable SSD streaming (toggle)
- Metal memory budget (slider, MB or % of RAM)
- Prefetch layers (number input)

### 10. panel sessions.ts
buildArgs: generate new CLI args. buildCommandPreview: show new args.

### 11. tests/test_ssd_streaming.py
Comprehensive tests for weight index, layer save/load/free, Metal memory verification, JANG round-trip, custom generate correctness, memory bounds.

## Prefill vs Decode Strategy

### Prefill (prompt processing):
- Load ALL layer weights into Metal memory
- Process prompt through all layers (standard mlx pattern)
- After prefill completes, free all layer weights
- KV cache is now populated with prompt context

### Decode (token generation):
- For each token:
  1. embed_tokens(token) — stays loaded permanently
  2. For layer i = 0 to N:
     a. Load layer i weights from safetensors
     b. h = layer_i(h, mask, cache=cache[i])
     c. Force synchronous execution of h
     d. Free layer i weights (replace with zeros, gc)
  3. norm(h) + lm_head(h) — stay loaded permanently
  4. Sample next token

### Performance estimates:
- SSD: ~7.4 GB/s (M4 internal SSD)
- 18GB model / 40 layers = 450MB per layer
- Load 1 layer: 450MB / 7.4 = 61ms
- All layers (1 token): 40 x 61ms = 2.4s -> ~0.4 tok/s
- With prefetching: ~0.8 tok/s
- 100GB model / 80 layers: ~0.07 tok/s (slow but RUNS)

## User-Controllable Settings (No Presets)

Per users explicit request: NO auto-detection, NO forced rules. Users control everything:

1. SSD Streaming toggle — on/off
2. Metal Memory Budget — slider (MB or % of RAM)
3. Prefetch Layers — number input (0 = on-demand, 1+ = prefetch ahead)
4. Cache behavior — users choose independently (SSD streaming does NOT force-disable caching)

## Compatibility

Works with: LLM, MoE, hybrid SSM, VLM, JANG, tool calling, reasoning, streaming SSE, Anthropic API, sleep/wake, image gen, embeddings, MCP tools.

Known limitations:
- Continuous batching: max_num_seqs should be 1
- Speculative decoding: incompatible (warn user)
- Prefill: all weights needed simultaneously
- Decode speed: limited by SSD bandwidth

## Old StreamingLayerWrapper — TO BE REMOVED

The old approach (sync barrier per layer + reduced wired limit + lock) does NOT work. Metal keeps weight buffers alive regardless. All code for StreamingLayerWrapper, lock_wired_limit, compute_streaming_wired_limit to be replaced.

## Open Questions

1. Async prefetching — load layer N+1 from SSD while layer N computes on GPU?
2. Partial MoE loading — load only active experts per token?
3. Persistent layer cache — LRU of recently-used layers via mmap?
4. JANG temp file location — model cache dir or system temp?
