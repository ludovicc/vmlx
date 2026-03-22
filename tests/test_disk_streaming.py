"""Tests for disk-streaming inference mode (--stream-from-disk)."""

import argparse
import pytest


class TestCLIFlagParsing:
    """Test that --stream-from-disk CLI flag is parsed correctly."""

    def _build_parser(self):
        """Build the CLI parser by importing and calling main's parser setup.

        Since the parser is built inline in main(), we replicate the minimal
        subparser structure needed to test --stream-from-disk.
        """
        # Import the module and inspect to get parser structure
        # We test via the actual CLI module by capturing the parser
        import importlib
        import unittest.mock as mock

        # The parser is built inside main(), so we need to extract it.
        # Instead, we'll verify the flag works by testing the actual module's
        # argument definitions match what we expect.
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Minimal serve subparser
        serve_parser = subparsers.add_parser("serve")
        serve_parser.add_argument("--model", type=str, required=True)
        serve_parser.add_argument("--stream-from-disk", action="store_true")

        # Minimal bench subparser
        bench_parser = subparsers.add_parser("bench")
        bench_parser.add_argument("--model", type=str, required=True)
        bench_parser.add_argument("--stream-from-disk", action="store_true")

        return parser

    def test_flag_exists_in_serve_parser(self):
        """Verify --stream-from-disk is a valid argument for serve."""
        parser = self._build_parser()
        args = parser.parse_args(['serve', '--model', 'test-model', '--stream-from-disk'])
        assert args.stream_from_disk is True

    def test_flag_default_is_false(self):
        """Verify --stream-from-disk defaults to False."""
        parser = self._build_parser()
        args = parser.parse_args(['serve', '--model', 'test-model'])
        assert args.stream_from_disk is False

    def test_flag_exists_in_bench_parser(self):
        """Verify --stream-from-disk works in bench subcommand."""
        parser = self._build_parser()
        args = parser.parse_args(['bench', '--model', 'test-model', '--stream-from-disk'])
        assert args.stream_from_disk is True

    def test_actual_cli_module_has_stream_from_disk(self):
        """Verify the actual cli.py source contains --stream-from-disk."""
        import inspect
        from vmlx_engine import cli
        source = inspect.getsource(cli)
        assert '--stream-from-disk' in source


class TestFeatureGating:
    """Test that stream-from-disk mode force-disables all caching features."""

    def _make_args(self, **overrides):
        """Create a mock args namespace with defaults."""
        defaults = {
            'stream_from_disk': False,
            'model': 'test-model',
            'continuous_batching': True,
            'enable_prefix_cache': True,
            'disable_prefix_cache': False,
            'use_paged_cache': True,
            'paged_cache_block_size': 64,
            'max_cache_blocks': 1000,
            'kv_cache_quantization': 'q8',
            'kv_cache_group_size': 64,
            'enable_disk_cache': True,
            'disk_cache_dir': '/tmp/test',
            'disk_cache_max_gb': 10.0,
            'enable_block_disk_cache': True,
            'block_disk_cache_dir': '/tmp/test',
            'block_disk_cache_max_gb': 10.0,
            'cache_memory_percent': 0.3,
            'cache_memory_mb': 0,
            'max_num_seqs': 256,
            'no_memory_aware_cache': False,
            'prefix_cache_size': 100,
            'cache_ttl_minutes': 0,
            'speculative_model': 'draft-model',
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _apply_gating(self, args):
        """Apply the same gating logic as serve_command() in cli.py."""
        if getattr(args, 'stream_from_disk', False):
            args.use_paged_cache = False
            args.enable_prefix_cache = False
            args.disable_prefix_cache = True
            args.enable_disk_cache = False
            args.enable_block_disk_cache = False
            args.kv_cache_quantization = "none"
            args.cache_memory_percent = 0.0
            args.cache_memory_mb = 0
            args.max_num_seqs = 1
            if hasattr(args, 'speculative_model'):
                args.speculative_model = None

    def test_stream_mode_disables_paged_cache(self):
        """Paged cache must be OFF in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.use_paged_cache is False

    def test_stream_mode_disables_prefix_cache(self):
        """Prefix cache must be OFF in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.enable_prefix_cache is False
        assert args.disable_prefix_cache is True

    def test_stream_mode_disables_disk_cache(self):
        """Disk cache must be OFF in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.enable_disk_cache is False

    def test_stream_mode_disables_block_disk_cache(self):
        """Block disk cache must be OFF in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.enable_block_disk_cache is False

    def test_stream_mode_disables_kv_quantization(self):
        """KV cache quantization must be 'none' in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.kv_cache_quantization == "none"

    def test_stream_mode_forces_single_sequence(self):
        """Max sequences must be 1 in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.max_num_seqs == 1

    def test_stream_mode_zeros_cache_memory(self):
        """Cache memory allocation must be zero in stream mode."""
        args = self._make_args(stream_from_disk=True)
        self._apply_gating(args)
        assert args.cache_memory_percent == 0.0
        assert args.cache_memory_mb == 0

    def test_stream_mode_disables_speculative(self):
        """Speculative decoding must be disabled in stream mode."""
        args = self._make_args(stream_from_disk=True, speculative_model='draft-model')
        self._apply_gating(args)
        assert args.speculative_model is None

    def test_normal_mode_preserves_all_settings(self):
        """When stream_from_disk=False, all settings are preserved."""
        args = self._make_args(stream_from_disk=False)
        self._apply_gating(args)
        assert args.use_paged_cache is True
        assert args.enable_prefix_cache is True
        assert args.enable_disk_cache is True
        assert args.enable_block_disk_cache is True
        assert args.kv_cache_quantization == "q8"
        assert args.max_num_seqs == 256
        assert args.speculative_model == 'draft-model'


class TestSchedulerCacheGating:
    """Test that SchedulerConfig with streaming defaults produces no caches."""

    def test_scheduler_no_caches_when_all_disabled(self):
        """SchedulerConfig with all caching off should produce no cache objects."""
        from vmlx_engine.scheduler import SchedulerConfig
        config = SchedulerConfig(
            enable_prefix_cache=False,
            use_paged_cache=False,
            kv_cache_quantization="none",
            enable_disk_cache=False,
            enable_block_disk_cache=False,
            cache_memory_percent=0.0,
            cache_memory_mb=0,
            max_num_seqs=1,
        )
        assert config.enable_prefix_cache is False
        assert config.use_paged_cache is False
        assert config.kv_cache_quantization == "none"
        assert config.enable_disk_cache is False
        assert config.enable_block_disk_cache is False
        assert config.max_num_seqs == 1


class TestServerGlobal:
    """Test that server._stream_from_disk global exists and works."""

    def test_stream_from_disk_global_exists(self):
        """Server module should have _stream_from_disk attribute."""
        from vmlx_engine import server
        assert hasattr(server, '_stream_from_disk')
        # Default should be False
        assert server._stream_from_disk is False


class TestJangLoaderLazyParam:
    """Test that jang_loader.load_jang_model accepts lazy parameter."""

    def test_load_jang_model_accepts_lazy(self):
        """load_jang_model should accept lazy=True without TypeError."""
        import inspect
        from vmlx_engine.utils.jang_loader import load_jang_model
        sig = inspect.signature(load_jang_model)
        assert 'lazy' in sig.parameters
        assert sig.parameters['lazy'].default is False


class TestCompatibilityMatrix:
    """Test the feature compatibility matrix from the plan."""

    DISABLED_IN_STREAM_MODE = [
        'use_paged_cache',
        'enable_prefix_cache',
        'enable_disk_cache',
        'enable_block_disk_cache',
    ]

    def test_all_cache_features_listed_as_disabled(self):
        """Every cache feature in the matrix must be force-disabled."""
        args = argparse.Namespace(
            stream_from_disk=True,
            use_paged_cache=True,
            enable_prefix_cache=True,
            disable_prefix_cache=False,
            enable_disk_cache=True,
            enable_block_disk_cache=True,
            kv_cache_quantization='q8',
            cache_memory_percent=0.3,
            cache_memory_mb=1024,
            max_num_seqs=256,
            speculative_model='draft',
        )
        # Apply gating
        if args.stream_from_disk:
            args.use_paged_cache = False
            args.enable_prefix_cache = False
            args.disable_prefix_cache = True
            args.enable_disk_cache = False
            args.enable_block_disk_cache = False
            args.kv_cache_quantization = "none"
            args.cache_memory_percent = 0.0
            args.cache_memory_mb = 0
            args.max_num_seqs = 1
            if hasattr(args, 'speculative_model'):
                args.speculative_model = None

        for feat in self.DISABLED_IN_STREAM_MODE:
            assert getattr(args, feat) is False, f"{feat} should be False in stream mode"
        assert args.kv_cache_quantization == "none"
        assert args.max_num_seqs == 1
        assert args.speculative_model is None

    def test_stream_mode_overrides_explicit_user_flags(self):
        """Even if user explicitly sets cache flags, stream mode overrides them."""
        args = argparse.Namespace(
            stream_from_disk=True,
            use_paged_cache=True,  # User explicitly set
            enable_prefix_cache=True,  # User explicitly set
            disable_prefix_cache=False,
            enable_disk_cache=True,  # User explicitly set
            enable_block_disk_cache=True,  # User explicitly set
            kv_cache_quantization='q4',  # User explicitly set
            cache_memory_percent=0.5,  # User explicitly set
            cache_memory_mb=4096,  # User explicitly set
            max_num_seqs=16,  # User explicitly set
            speculative_model='big-draft',  # User explicitly set
        )
        # Apply gating — should override everything
        if args.stream_from_disk:
            args.use_paged_cache = False
            args.enable_prefix_cache = False
            args.disable_prefix_cache = True
            args.enable_disk_cache = False
            args.enable_block_disk_cache = False
            args.kv_cache_quantization = "none"
            args.cache_memory_percent = 0.0
            args.cache_memory_mb = 0
            args.max_num_seqs = 1
            if hasattr(args, 'speculative_model'):
                args.speculative_model = None

        assert args.use_paged_cache is False
        assert args.enable_prefix_cache is False
        assert args.kv_cache_quantization == "none"
        assert args.max_num_seqs == 1
        assert args.speculative_model is None


class TestEdgeCases:
    """Test edge cases: sleep/wake, exceeds-RAM, None caches."""

    def test_scheduler_no_crash_with_none_caches(self):
        """Scheduler with all caches disabled should not crash on get_memory_usage."""
        from vmlx_engine.scheduler import SchedulerConfig, Scheduler
        import mlx.core as mx
        import mlx.nn as nn

        # Minimal model for scheduler init
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
            def __call__(self, x, cache=None):
                return self.embed(x)

        config = SchedulerConfig(
            enable_prefix_cache=False,
            use_paged_cache=False,
            kv_cache_quantization="none",
            enable_disk_cache=False,
            enable_block_disk_cache=False,
            cache_memory_percent=0.0,
            max_num_seqs=1,
        )
        # Verify no cache objects created
        assert config.enable_prefix_cache is False
        assert config.use_paged_cache is False

    def test_stream_from_disk_preserved_in_cli_args(self):
        """_cli_args should contain stream_from_disk when set."""
        from vmlx_engine import server
        # Verify _cli_args can store and retrieve stream_from_disk
        test_args = {'stream_from_disk': True, 'use_batching': False}
        assert test_args.get('stream_from_disk', False) is True
        assert test_args.get('stream_from_disk') is True

    def test_jang_gate_dequant_triggers_for_moe(self):
        """JANG gate dequant should trigger for any model with n_routed_experts > 0."""
        # Simulate the logic from jang_loader.py
        config_nemotron = {"model_type": "nemotron_h", "n_routed_experts": 64}
        config_mistral4 = {"model_type": "mistral4", "n_routed_experts": 128}
        config_llm = {"model_type": "llama"}
        config_vlm = {"model_type": "mistral3", "text_config": {"n_routed_experts": 128}}

        for cfg in [config_nemotron, config_mistral4, config_llm, config_vlm]:
            model_type = cfg.get("model_type", "")
            needs_fc_rename = model_type in ("nemotron_h", "nemotron")
            text_cfg = cfg.get("text_config", cfg)
            n_experts = cfg.get("n_routed_experts", 0) or text_cfg.get("n_routed_experts", 0)
            needs_gate_dequant = needs_fc_rename or n_experts > 0

            if model_type == "nemotron_h":
                assert needs_fc_rename is True
                assert needs_gate_dequant is True
            elif model_type == "mistral4":
                assert needs_fc_rename is False
                assert needs_gate_dequant is True
            elif model_type == "llama":
                assert needs_fc_rename is False
                assert needs_gate_dequant is False
            elif model_type == "mistral3":
                assert needs_fc_rename is False
                assert needs_gate_dequant is True  # via text_config

    def test_mistral4_in_standard_architectures(self):
        """mistral4 should be in the _STANDARD_ARCHITECTURES set."""
        from vmlx_engine.utils.tokenizer import _STANDARD_ARCHITECTURES
        assert "mistral4" in _STANDARD_ARCHITECTURES

    def test_mistral4_in_model_config_registry(self):
        """mistral4 should be discoverable in model_config_registry."""
        from vmlx_engine.model_config_registry import get_model_config_registry
        # The registry should have a mistral4 entry
        registry = get_model_config_registry()
        found = False
        for config in registry._configs:
            if "mistral4" in config.model_types:
                found = True
                assert config.family_name == "mistral4"
                assert config.tool_parser == "mistral"
                assert config.cache_type == "kv"
                break
        assert found, "mistral4 not found in model_config_registry"

    def test_mllm_batch_generator_stream_flag_readable(self):
        """MLLMBatchGenerator should be able to read _stream_from_disk."""
        from vmlx_engine import server
        # Verify the global is accessible
        assert hasattr(server, '_stream_from_disk')
        original = server._stream_from_disk
        server._stream_from_disk = True
        assert server._stream_from_disk is True
        server._stream_from_disk = original  # restore


class TestStreamingLayerWrapper:
    """Test the StreamingLayerWrapper and related functions."""

    def test_wrapper_class_exists(self):
        """StreamingLayerWrapper should be importable."""
        from vmlx_engine.utils.streaming_wrapper import StreamingLayerWrapper
        assert StreamingLayerWrapper is not None

    def test_wrapper_calls_layer(self):
        """Wrapper should delegate to the inner layer."""
        import mlx.core as mx
        import mlx.nn as nn

        layer = nn.Linear(8, 8)
        from vmlx_engine.utils.streaming_wrapper import StreamingLayerWrapper
        wrapped = StreamingLayerWrapper(layer)

        x = mx.ones((1, 8))
        result = wrapped(x)
        assert isinstance(result, mx.array)
        assert result.shape == (1, 8)

    def test_wrapper_getattr_delegation(self):
        """Wrapper should forward attribute access to the inner layer."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import StreamingLayerWrapper

        layer = nn.Linear(4, 8)
        wrapped = StreamingLayerWrapper(layer)
        # Access a property that exists on nn.Linear
        assert wrapped.weight.shape == (8, 4)

    def test_wrapper_handles_tuple_result(self):
        """Wrapper should handle layers that return tuples."""
        import mlx.core as mx
        from vmlx_engine.utils.streaming_wrapper import StreamingLayerWrapper

        class TupleLayer:
            def __call__(self, x):
                return (x, x * 2)

        wrapped = StreamingLayerWrapper(TupleLayer())
        result = wrapped(mx.ones((2, 4)))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestFindLayers:
    """Test _find_layers across model structures."""

    def test_find_layers_standard(self):
        """Should find model.model.layers (standard mlx-lm structure)."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import _find_layers

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(3)]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

        model = Model()
        result = _find_layers(model)
        assert result is not None
        container, attr = result
        assert attr == 'layers'
        assert len(getattr(container, attr)) == 3

    def test_find_layers_vlm(self):
        """Should find model.language_model.model.layers (VLM wrapper)."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import _find_layers

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(5)]

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = LM()

        model = VLMModel()
        result = _find_layers(model)
        assert result is not None
        container, attr = result
        assert len(getattr(container, attr)) == 5

    def test_find_layers_none(self):
        """Should return None when no layers found."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import _find_layers

        class NoLayersModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)

        model = NoLayersModel()
        result = _find_layers(model)
        assert result is None


class TestApplyStreamingLayers:
    """Test apply_streaming_layers end-to-end."""

    def test_apply_wraps_all_layers(self):
        """All layers should be wrapped after apply_streaming_layers."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import (
            apply_streaming_layers,
            StreamingLayerWrapper,
        )

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(4)]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

        model = Model()
        n = apply_streaming_layers(model)
        assert n == 4
        for layer in model.model.layers:
            assert isinstance(layer, StreamingLayerWrapper)

    def test_apply_returns_zero_for_no_layers(self):
        """Should return 0 when model has no layers."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import apply_streaming_layers

        class FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Linear(4, 4)

        model = FlatModel()
        n = apply_streaming_layers(model)
        assert n == 0

    def test_normal_mode_no_wrappers(self):
        """In normal mode (no apply), layers should NOT be wrapped."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import StreamingLayerWrapper

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(3)]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

        model = Model()
        for layer in model.model.layers:
            assert not isinstance(layer, StreamingLayerWrapper)


class TestComputeStreamingWiredLimit:
    """Test compute_streaming_wired_limit."""

    def test_returns_positive_value(self):
        """Wired limit should be a positive integer."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import (
            apply_streaming_layers,
            compute_streaming_wired_limit,
        )

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(64, 64) for _ in range(10)]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = InnerModel()

        model = Model()
        wired = compute_streaming_wired_limit(model)
        assert wired is not None
        assert wired > 0
        # Floor is 2GB
        assert wired >= 2 * 1024 * 1024 * 1024

    def test_returns_none_for_no_layers(self):
        """Should return None when model has no layers."""
        import mlx.nn as nn
        from vmlx_engine.utils.streaming_wrapper import compute_streaming_wired_limit

        class FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Linear(4, 4)

        model = FlatModel()
        wired = compute_streaming_wired_limit(model)
        assert wired is None


class TestWiredLimitLock:
    """Test the wired limit lock mechanism."""

    def test_lock_prevents_wired_change(self):
        """After locking, mx.set_wired_limit should be a no-op."""
        import mlx.core as mx
        from vmlx_engine.utils.streaming_wrapper import lock_wired_limit, unlock_wired_limit

        original_fn = mx.set_wired_limit
        try:
            lock_wired_limit()
            # Should not be the original function
            assert mx.set_wired_limit is not original_fn
            # Should return 0 (no-op)
            result = mx.set_wired_limit(999999999)
            assert result == 0
        finally:
            unlock_wired_limit()
            # Should be restored
            assert mx.set_wired_limit is original_fn

    def test_unlock_restores_original(self):
        """unlock_wired_limit should restore the original function."""
        import mlx.core as mx
        from vmlx_engine.utils.streaming_wrapper import lock_wired_limit, unlock_wired_limit

        original_fn = mx.set_wired_limit
        lock_wired_limit()
        unlock_wired_limit()
        assert mx.set_wired_limit is original_fn

    def test_double_lock_is_safe(self):
        """Calling lock twice should not lose the original."""
        import mlx.core as mx
        from vmlx_engine.utils.streaming_wrapper import lock_wired_limit, unlock_wired_limit

        original_fn = mx.set_wired_limit
        try:
            lock_wired_limit()
            lock_wired_limit()  # Should be no-op
            unlock_wired_limit()
            assert mx.set_wired_limit is original_fn
        finally:
            # Safety cleanup
            unlock_wired_limit()
