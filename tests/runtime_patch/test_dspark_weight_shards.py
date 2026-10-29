# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only draft shard tests, also runnable with stdlib unittest.

Set VLLM_TEST_UPSTREAM_ROOT to the paired vLLM checkout to exercise its actual
loader methods and index validation with tensor I/O mocked out. No vLLM/torch
import or accelerator initialization is needed.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import glob
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterable, cast
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
CORE_FIX = REPO / "vllm_hcu/patch/worker/core_fix"


def _load_adapter() -> ModuleType:
    # core_fix's public package imports accelerator adapters. Load only this
    # adapter and its real dependency-light _common module in a private package.
    package_name = "_hcu_dspark_shard_tests"
    package = ModuleType(package_name)
    package.__path__ = [str(CORE_FIX)]
    name = f"{package_name}.patch_dspark_weight_shards"
    spec = importlib.util.spec_from_file_location(
        name, CORE_FIX / "patch_dspark_weight_shards.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package_name: package, name: module}):
        spec.loader.exec_module(module)
    return module


ADAPTER = _load_adapter()


def _draft():
    class DSparkDeepseekV4ForCausalLM:
        __module__ = ADAPTER._DRAFT_MODULE

        def __init__(self):
            self.received = []

        def _remap_dspark_name(self, name):
            return name if re.match(r"mtp\.\d+\.", name) else None

        def load_weights(self, weights):
            self.received = [
                name for name, _ in weights if self._remap_dspark_name(name) is not None
            ]
            return set(self.received)

    return DSparkDeepseekV4ForCausalLM()


def _config(folder):
    return SimpleNamespace(
        model=str(folder),
        revision="draft-revision",
        quantization=None,
        hf_config=SimpleNamespace(model_type="deepseek_v41"),
    )


def _fake_loader(prepared):
    class DefaultModelLoader:
        def __init__(self):
            self.load_config = SimpleNamespace(load_format="safetensors")
            self.calls = []
            self.during_load = None

        def _prepare_weights(
            self,
            model_name_or_path,
            subfolder,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        ):
            self.calls.append(
                (
                    model_name_or_path,
                    subfolder,
                    revision,
                    fall_back_to_pt,
                    allow_patterns_overrides,
                )
            )
            return prepared

        def load_weights(self, model, model_config):
            if self.during_load is not None:
                self.during_load(model)
            return self._prepare_weights(
                model_config.model, None, model_config.revision, True, None
            )

    module = ModuleType(ADAPTER.TARGET_MODULE)
    module.DefaultModelLoader = DefaultModelLoader
    ADAPTER.apply_to_module(module)
    return DefaultModelLoader(), module


class ShardFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dspark-shard-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.weight_map = {
            "embed.weight": "target.safetensors",
            "head.weight": "target.safetensors",
            "layers.0.attn.wq_a.weight": "backbone.safetensors",
            "mtp.0.main_proj.weight": "context.safetensors",
            "mtp.0.main_norm.weight": "context.safetensors",
            "mtp.0.attn.wq_a.weight_scale": "stage-c.safetensors",
            "mtp.0.ffn.shared_experts.w1.weight": "stage-c.safetensors",
            "mtp.1.ffn.experts.0.w1.weight_packed": "stage-a.safetensors",
            "mtp.1.ffn.experts.0.w1.weight_scale_inv": "stage-a.safetensors",
            "mtp.2.norm.weight": "head [draft].safetensors",
            "mtp.2.markov_head.embed.weight": "head [draft].safetensors",
            "mtp.2.markov_head.head.weight": "head [draft].safetensors",
            "mtp.2.confidence_head.proj.weight": "confidence.safetensors",
        }
        self.files = [
            str(self.folder / filename)
            for filename in dict.fromkeys(self.weight_map.values())
        ]
        self.index_path = self.folder / "model.safetensors.index.json"
        self.write_index({"weight_map": self.weight_map})
        self.config = _config(self.folder)
        self.draft = _draft()

    def write_index(self, index):
        self.index_path.write_text(json.dumps(index), encoding="utf-8")


class DraftShardTests(ShardFixture):
    def test_all_accepted_weights_choose_dynamic_shards_in_original_order(self):
        selected = ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        expected = self.files[2:]
        self.assertEqual(selected, expected)
        accepted = {
            name
            for name, filename in self.weight_map.items()
            if str(self.folder / filename) in selected
            and self.draft._remap_dspark_name(name) is not None
        }
        self.assertEqual(accepted, {n for n in self.weight_map if n.startswith("mtp.")})
        self.assertEqual(len(selected), len(set(selected)))

    def test_selection_defers_to_remapper_for_shared_or_future_aliases(self):
        self.weight_map["shared_alias.weight_scale"] = "target.safetensors"
        self.write_index({"weight_map": self.weight_map})
        original = self.draft._remap_dspark_name
        self.draft._remap_dspark_name = lambda name: (
            "model.alias" if name == "shared_alias.weight_scale" else original(name)
        )
        selected = ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        self.assertIn(self.files[0], selected)
        self.assertNotIn(self.files[1], selected)

    def test_disabled_optional_confidence_head_does_not_require_its_shard(self):
        original = self.draft._remap_dspark_name
        self.draft._remap_dspark_name = lambda name: (
            None if "confidence_head." in name else original(name)
        )
        selected = ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        self.assertNotIn(str(self.folder / "confidence.safetensors"), selected)

    def test_missing_or_invalid_index_preserves_prepared_files(self):
        self.index_path.rename(self.folder / "saved-index.json")
        self.assertIsNone(
            ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        )
        for invalid in (
            "{broken",
            "[]",
            "{}",
            '{"weight_map": []}',
            '{"weight_map": {"mtp.0.weight": 1}}',
        ):
            with self.subTest(index=invalid):
                self.index_path.write_text(invalid, encoding="utf-8")
                selected = ADAPTER._select_draft_files(
                    self.draft, str(self.folder), self.files
                )
                self.assertIsNone(selected)

    def test_unavailable_draft_shard_or_no_accepted_tensors_falls_back(self):
        self.assertIsNone(
            ADAPTER._select_draft_files(self.draft, str(self.folder), self.files[:-1])
        )
        self.write_index({"weight_map": {"embed.weight": "target.safetensors"}})
        self.assertIsNone(
            ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        )

    def test_main_model_and_config_remain_isolated(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        before_config = dict(vars(self.config))
        before_loader = set(vars(loader))
        result = loader.load_weights(self.draft, self.config)
        self.assertEqual(result[1], self.files[2:])
        self.assertIs(loader.load_weights(object(), self.config), prepared)
        self.assertEqual(vars(self.config), before_config)
        self.assertEqual(set(vars(loader)), before_loader)
        self.assertEqual(loader.calls[0], loader.calls[1])
        self.assertIsNone(ADAPTER._DRAFT_LOAD.get())

    def test_other_models_and_existing_sources_preserve_preparation(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        for attribute, value in (
            ("allow_patterns_overrides", ["custom*.safetensors"]),
            ("secondary_weights", [object()]),
        ):
            with self.subTest(attribute=attribute):
                model = _draft()
                setattr(model, attribute, value)
                self.assertIs(loader.load_weights(model, self.config), prepared)
        self.config.hf_config.model_type = "deepseek_v4"
        self.assertIs(loader.load_weights(self.draft, self.config), prepared)

    def test_non_safetensors_or_inapplicable_formats_are_unchanged(self):
        prepared = (str(self.folder), self.files, False)
        loader, _ = _fake_loader(prepared)
        self.assertIs(loader.load_weights(self.draft, self.config), prepared)
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        for load_format in ("pt", "npcache", "mistral", "custom"):
            loader.load_config.load_format = load_format
            self.assertIs(loader.load_weights(self.draft, self.config), prepared)

    def test_secondary_path_and_subfolder_arguments_are_unchanged(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        pattern = ["custom.safetensors"]
        observed = []

        def inspect_sources(model):
            del model
            for path, subfolder in (
                ("secondary-model", None),
                (self.config.model, "sub"),
            ):
                observed.append(
                    loader._prepare_weights(path, subfolder, "rev2", False, pattern)
                )

        loader.during_load = inspect_sources
        loader.load_weights(self.draft, self.config)
        self.assertTrue(all(result is prepared for result in observed))
        self.assertEqual(
            loader.calls[0], ("secondary-model", None, "rev2", False, pattern)
        )
        self.assertEqual(
            loader.calls[1], (self.config.model, "sub", "rev2", False, pattern)
        )

    def test_nested_main_load_is_not_filtered_and_restores_draft_scope(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        observed = []

        def nested(model):
            if model is self.draft:
                observed.append(loader.load_weights(object(), self.config))

        loader.during_load = nested
        result = loader.load_weights(self.draft, self.config)
        self.assertIs(observed[0], prepared)
        self.assertEqual(result[1], self.files[2:])
        self.assertIsNone(ADAPTER._DRAFT_LOAD.get())

    def test_exception_restores_context_and_keeps_original_failure(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)

        def fail(model):
            del model
            raise FileNotFoundError("original checkpoint validation failure")

        loader.during_load = fail
        with self.assertRaisesRegex(FileNotFoundError, "original checkpoint"):
            loader.load_weights(self.draft, self.config)
        self.assertIsNone(ADAPTER._DRAFT_LOAD.get())
        loader.during_load = None
        self.assertIs(loader.load_weights(object(), self.config), prepared)

    def test_concurrent_main_load_on_same_loader_keeps_its_full_file_list(self):
        prepared = (str(self.folder), self.files, True)
        loader, _ = _fake_loader(prepared)
        barrier = threading.Barrier(2)
        loader.during_load = lambda model: barrier.wait(timeout=5)
        with ThreadPoolExecutor(max_workers=2) as executor:
            draft = executor.submit(loader.load_weights, self.draft, self.config)
            main = executor.submit(loader.load_weights, object(), self.config)
            self.assertEqual(draft.result()[1], self.files[2:])
            self.assertIs(main.result(), prepared)
        self.assertIsNone(ADAPTER._DRAFT_LOAD.get())

    def test_patch_is_idempotent_and_detects_stale_wrapper(self):
        loader, module = _fake_loader((str(self.folder), self.files, True))
        self.assertFalse(ADAPTER.apply_to_module(module))
        type(loader)._prepare_weights = lambda *args: None
        with self.assertRaises(ADAPTER.PatchCompatibilityError):
            ADAPTER.apply_to_module(module)


def _upstream_fixture(root: Path, weight_map: dict[str, str], opened: list[str]):
    """Execute the paired loader's real AST; replace only imports/tensor I/O."""
    loader_path = root / "vllm/model_executor/model_loader/default_loader.py"
    utils_path = root / "vllm/model_executor/model_loader/weight_utils.py"
    loader_tree = ast.parse(loader_path.read_text(encoding="utf-8"))
    utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    loader_class = next(
        node
        for node in loader_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DefaultModelLoader"
    )
    filter_function = next(
        node
        for node in utils_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "filter_duplicate_safetensors_files"
    )

    class BaseModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    def tensor_iterator(files, *args, **kwargs):
        del args, kwargs
        for filename in files:
            opened.append(filename)
            for name, shard in weight_map.items():
                if str(Path(filename).parent / shard) == filename:
                    yield name, object()

    module = ModuleType(ADAPTER.TARGET_MODULE)
    module.__dict__.update(
        BaseModelLoader=BaseModelLoader,
        dataclasses=dataclasses,
        glob=glob,
        os=os,
        time=time,
        json=json,
        Iterable=Iterable,
        cast=cast,
        SAFE_WEIGHTS_INDEX_NAME="model.safetensors.index.json",
        maybe_download_from_modelscope=lambda *args: None,
        list_filtered_repo_files=lambda **kwargs: [
            file
            for pattern in kwargs["allow_patterns"]
            for file in glob.glob(os.path.join(kwargs["model_name_or_path"], pattern))
        ],
        instrument=lambda **kwargs: lambda function: function,
        logger=SimpleNamespace(info_once=lambda *args: None),
        safetensors_weights_iterator=tensor_iterator,
        multi_thread_safetensors_weights_iterator=tensor_iterator,
        fastsafetensors_weights_iterator=tensor_iterator,
        instanttensor_weights_iterator=tensor_iterator,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(
        ast.Module(body=[future, filter_function, loader_class], type_ignores=[])
    )
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(loader_path), "exec"), module.__dict__)
    # Expert ownership is unrelated to file selection and needs distributed state.
    module.DefaultModelLoader._init_ep_weight_filter = lambda *args: None
    return module


@unittest.skipUnless(
    os.environ.get("VLLM_TEST_UPSTREAM_ROOT"),
    "set VLLM_TEST_UPSTREAM_ROOT for paired-loader CPU integration",
)
class PairedLoaderTests(ShardFixture):
    def upstream_loader(self, load_format="safetensors", multithread=False):
        for filename in self.files:
            Path(filename).touch()
        opened = []
        module = _upstream_fixture(
            Path(os.environ["VLLM_TEST_UPSTREAM_ROOT"]), self.weight_map, opened
        )
        config = SimpleNamespace(
            load_format=load_format,
            model_loader_extra_config={
                "enable_weights_track": False,
                "enable_multithread_load": multithread,
            },
            safetensors_load_strategy=None,
            use_tqdm_on_load=False,
            safetensors_prefetch_num_threads=1,
            safetensors_prefetch_block_size=1,
        )
        return module, module.DefaultModelLoader(config), opened

    def test_paired_loader_reads_only_draft_shards_and_keeps_all_accepted_weights(self):
        for load_format in (
            "auto", "hf", "safetensors", "fastsafetensors", "instanttensor"
        ):
            with self.subTest(load_format=load_format):
                module, loader, opened = self.upstream_loader(load_format)
                loader.load_weights(self.draft, self.config)
                baseline = set(opened)
                self.assertEqual(baseline, set(self.files))
                opened.clear()
                ADAPTER.apply_to_module(module)
                loader.load_weights(self.draft, self.config)
                self.assertEqual(set(opened), set(self.files[2:]))
                self.assertEqual(len(opened), len(set(opened)))
                self.assertEqual(
                    set(self.draft.received),
                    {n for n in self.weight_map if n.startswith("mtp.")},
                )
                opened.clear()
                main = _draft()
                type(main).__module__ = "test.target_model"
                loader.load_weights(main, self.config)
                self.assertEqual(set(opened), baseline)

    def test_paired_loader_multithread_uses_the_same_selected_files(self):
        module, loader, opened = self.upstream_loader(multithread=True)
        ADAPTER.apply_to_module(module)
        loader.load_weights(self.draft, self.config)
        self.assertEqual(set(opened), set(self.files[2:]))

    def test_paired_loader_missing_index_keeps_original_iteration(self):
        module, loader, opened = self.upstream_loader()
        self.index_path.rename(self.folder / "saved-index.json")
        ADAPTER.apply_to_module(module)
        loader.load_weights(self.draft, self.config)
        self.assertEqual(set(opened), set(self.files))

    def test_paired_loader_index_validation_precedes_filtering(self):
        module, loader, opened = self.upstream_loader()
        with self.assertRaisesRegex(FileNotFoundError, "referenced in index") as error:
            loader._prepare_weights(
                self.config.model,
                None,
                None,
                True,
                ["context.safetensors", "stage-c.safetensors"],
            )
        # The second exact pattern is never reached after the first matches.
        self.assertIn(str(self.folder / "stage-c.safetensors"), str(error.exception))
        ADAPTER.apply_to_module(module)
        # Removing an unrelated target shard must retain upstream validation.
        Path(self.files[0]).rename(self.folder / "target.backup")
        with self.assertRaisesRegex(FileNotFoundError, "referenced in index"):
            loader.load_weights(self.draft, self.config)
        self.assertEqual(opened, [])

    def test_load_dspark_model_still_aliases_target_embedding_and_lm_head(self):
        root = Path(os.environ["VLLM_TEST_UPSTREAM_ROOT"])
        module, loader, opened = self.upstream_loader()
        ADAPTER.apply_to_module(module)
        draft = self.draft
        draft.has_own_embed_tokens = False
        draft.has_own_lm_head = False
        draft.model = SimpleNamespace(embed_tokens=object())
        draft.lm_head = object()
        self.config.get_vocab_size = lambda: 16
        self.config.hf_config.draft_vocab_size = None
        target = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object()), lm_head=object()
        )
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_model_config=self.config,
                draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
                attention_backend=None,
                kv_cache_dtype=None,
            ),
            model_config=SimpleNamespace(get_vocab_size=lambda: 16),
            parallel_config=object(),
            attention_config=SimpleNamespace(backend=None),
            cache_config=object(),
            load_config=loader.load_config,
        )

        def get_model(*, vllm_config, model_config):
            del vllm_config
            loader.load_weights(draft, model_config)
            return draft

        def fake_module(name, **attributes):
            result = ModuleType(name)
            result.__dict__.update(attributes)
            return result

        eagle = fake_module(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils",
            get_pp_group=lambda: SimpleNamespace(world_size=1),
            BaseLayerWithLoRA=type("BaseLayerWithLoRA", (), {}),
            PPMissingLayer=type("PPMissingLayer", (), {}),
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        eagle_source = root / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"
        eagle_tree = ast.parse(eagle_source.read_text(encoding="utf-8"))
        sharing_functions = [
            node
            for node in eagle_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in ("_should_share", "get_target_lm_head", "maybe_share_target_embed")
        ]
        code = ast.fix_missing_locations(
            ast.Module(body=[future, *sharing_functions], type_ignores=[])
        )
        exec(compile(code, str(eagle_source), "exec"), eagle.__dict__)
        dependencies = {
            eagle.__name__: eagle,
            "vllm.compilation.backends": fake_module(
                "vllm.compilation.backends",
                set_model_tag=lambda tag: contextlib.nullcontext(),
            ),
            "vllm.model_executor.model_loader": fake_module(
                "vllm.model_executor.model_loader", get_model=get_model
            ),
            "vllm.model_executor.models.qwen3_dflash": fake_module(
                "vllm.model_executor.models.qwen3_dflash",
                dflash_has_any_non_causal=lambda hf_config: False,
            ),
            "vllm.model_executor.models.utils": fake_module(
                "vllm.model_executor.models.utils",
                get_draft_quant_config=lambda cfg: None,
            ),
        }
        dspark_source = root / "vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
        dspark_tree = ast.parse(dspark_source.read_text(encoding="utf-8"))
        load_function = next(
            node for node in dspark_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_dspark_model"
        )
        namespace = {
            "replace": lambda obj, **changes: SimpleNamespace(
                **dict(vars(obj), **changes)
            ),
            "_get_dspark_parallel_config": lambda parallel, size: parallel,
            "_resolve_dspark_attention_backend": lambda *args: None,
            "get_pp_safe_draft_load_config": lambda cfg: cfg,
        }
        code = ast.fix_missing_locations(
            ast.Module(body=[future, load_function], type_ignores=[])
        )
        exec(compile(code, str(dspark_source), "exec"), namespace)
        with patch.dict(sys.modules, dependencies):
            result = namespace["load_dspark_model"](target, config)
        self.assertIs(result.model.embed_tokens, target.model.embed_tokens)
        self.assertIs(result.lm_head, target.lm_head)
        self.assertEqual(set(opened), set(self.files[2:]))
        self.assertFalse(hasattr(config, "quant_config"))
        self.assertFalse(hasattr(self.config, "allow_patterns_overrides"))

    @unittest.skipIf(
        sys.version_info < (3, 9), "official remapper uses str.removeprefix"
    )
    def test_official_remapper_covers_context_heads_experts_and_scale_aliases(self):
        root = Path(os.environ["VLLM_TEST_UPSTREAM_ROOT"])
        source = root / "vllm/models/deepseek_v41/amd/dspark.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DSparkDeepseekV4ForCausalLM"
        )
        mapper = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_remap_dspark_name"
        )
        namespace = {"re": re}
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        code = ast.fix_missing_locations(
            ast.Module(body=[future, mapper], type_ignores=[])
        )
        exec(compile(code, str(source), "exec"), namespace)
        self.draft.model = SimpleNamespace(confidence_head=object())
        self.draft._remap_dspark_name = namespace[mapper.name].__get__(self.draft)
        selected = ADAPTER._select_draft_files(self.draft, str(self.folder), self.files)
        self.assertEqual(selected, self.files[2:])
        self.assertEqual(
            self.draft._remap_dspark_name("mtp.2.markov_head.embed.weight"),
            "model.markov_head.markov_w1.weight",
        )
        self.assertEqual(
            self.draft._remap_dspark_name("mtp.2.markov_head.head.weight"),
            "model.markov_head.markov_w2.weight",
        )
        self.assertIsNone(self.draft._remap_dspark_name("embed.weight"))
        self.assertIsNone(self.draft._remap_dspark_name("head.weight"))


if __name__ == "__main__":
    unittest.main()
