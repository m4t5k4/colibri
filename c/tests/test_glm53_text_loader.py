"""Explicit text-only loader policy, with a dependency-free tiny checkpoint."""
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "glm53.c").read_text(encoding="utf-8")


def body(name):
    match = re.search(r"static\s+[^;{}]+\b" + name + r"\([^;{}]*\)\s*\{", SOURCE)
    assert match, name
    start = match.end()
    depth = 1
    for end in range(start, len(SOURCE)):
        depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
        if not depth:
            return SOURCE[start:end]
    raise AssertionError(name)


def fixture(directory):
    """One KDA/dense layer and one real, loadable vision block; no torch."""
    text = dict(hidden_size=4, num_hidden_layers=1, vocab_size=8,
                first_k_dense_replace=1, intermediate_size=8,
                layer_types=["linear_attention"],
                num_attention_heads=2, q_lora_rank=4, kv_lora_rank=2,
                qk_nope_head_dim=2, v_head_dim=2, n_routed_experts=1,
                num_experts_per_tok=1, moe_intermediate_size=32, hc_mult=1,
                linear_attn_config=dict(num_heads=2, head_dim=2,
                                       short_conv_kernel_size=2, kda_layers=[0]))
    vision = dict(depth=1, hidden_size=4, num_heads=2, intermediate_size=8,
                  patch_size=1, temporal_patch_size=1, spatial_merge_size=1,
                  in_channels=1, out_hidden_size=4, projection_intermediate_size=8)
    (directory / "config.json").write_text(json.dumps(dict(text_config=text,
                                                          vision_config=vision)))
    shapes = {}
    p = "model.language_model."
    shapes[p + "embed_tokens.weight"] = [8, 4]
    shapes[p + "norm.weight"] = [4]
    shapes["lm_head.weight"] = [8, 4]
    l = p + "layers.0."
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        shapes[l + name] = [4]
    for kind in ("attn", "ffn"):
        for suffix, shape in (("fn", [3, 4]), ("base", [3]), ("scale", [3])):
            shapes[l + "hc_" + kind + "_" + suffix] = shape
    a = l + "self_attn."
    for name in ("q", "k", "v", "o", "g_b", "f_b"):
        shapes[a + name + "_proj.weight"] = [4, 4] if name in ("q", "k", "v", "o") else [4, 2]
    for name in ("g_a", "f_a", "b"):
        shapes[a + name + "_proj.weight"] = [2, 4]
    for name, shape in (("dt_bias", [4]), ("A_log", [2]), ("o_norm.weight", [2])):
        shapes[a + name] = shape
    for name in ("q", "k", "v"):
        shapes[a + name + "_conv1d.weight"] = [4, 1, 2]
    for name in ("gate", "up", "down"):
        shapes[l + "mlp." + name + "_proj.weight"] = [4, 8] if name == "down" else [8, 4]
    v = "model.visual."
    for name, shape in (("patch_embed.proj.weight", [4, 1]),
                        ("patch_embed.proj.bias", [4]), ("post_layernorm.weight", [4]),
                        ("downsample.weight", [4, 4]), ("downsample.bias", [4]),
                        ("merger.proj.weight", [4, 4]),
                        ("merger.post_projection_norm.weight", [4]),
                        ("merger.post_projection_norm.bias", [4]),
                        ("merger.gate_proj.weight", [8, 4]),
                        ("merger.up_proj.weight", [8, 4]), ("merger.down_proj.weight", [4, 8])):
        shapes[v + name] = shape
    for name, shape in (("norm1.weight", [4]), ("norm2.weight", [4]),
                        ("attn.qkv.weight", [12, 4]), ("attn.qkv.bias", [12]),
                        ("attn.q_norm.weight", [2]), ("attn.k_norm.weight", [2]),
                        ("attn.proj.weight", [4, 4]), ("attn.proj.bias", [4]),
                        ("mlp.gate_proj.weight", [8, 4]), ("mlp.gate_proj.bias", [8]),
                        ("mlp.up_proj.weight", [8, 4]), ("mlp.up_proj.bias", [8]),
                        ("mlp.down_proj.weight", [4, 8]), ("mlp.down_proj.bias", [4])):
        shapes[v + "blocks.0." + name] = shape
    header, payload = {}, bytearray()
    for name, shape in shapes.items():
        begin = len(payload)
        payload.extend(struct.pack("<f", 0.125) * math.prod(shape))
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[begin, len(payload)])
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class TextLoaderPolicy(unittest.TestCase):
    def test_default_enables_vision(self):
        self.assertRegex(body("model_load_range"), r"model_load_range_ex\(\s*m,\s*dir,\s*layer_begin,\s*layer_end,\s*load_io,\s*1\s*\)")

    def test_adapters_explicitly_opt_out(self):
        for name, arguments in (("glm53_edge_engine_open", r"0,\s*0,\s*1,\s*0"),
                                ("glm53_segment_engine_open", r"begin,\s*end,\s*0,\s*0")):
            self.assertRegex(body(name), r"model_load_range_ex\(&engine->model,\s*options->model_dir,\s*" + arguments + r"\)")

    def test_normal_loading_uses_default(self):
        self.assertRegex(body("model_load"), r"model_load_range\(m,\s*dir,\s*0,\s*-1,\s*1\)")
        self.assertIn("model_load(&served, snap)", SOURCE)
        self.assertIn("model_load(&model, dir)", SOURCE)

    def test_only_explicit_parameter_controls_skip(self):
        loader = body("model_load_range_ex")
        self.assertRegex(loader, r"if\s*\(load_vision\)\s*vision_load\(m\)")
        self.assertEqual(loader.count("vision_load(m)"), 1)
        self.assertIn("load_cfg(&m->c, dir)", loader)
        self.assertNotRegex(SOURCE, r"getenv\([^)]*[Vv][Ii][Ss][Ii][Oo][Nn]")
        self.assertNotRegex(loader, r"m->c\.vis_\w+\s*=")

    def test_vision_still_loads_geometry_and_f32_weights(self):
        tower = body("vision_load")
        self.assertIn("c->vis_hidden / c->vis_heads", tower)
        self.assertIn("hidden * patch", tower)
        self.assertIn("out * hidden * merge * merge", tower)
        self.assertIn("3 * hidden * hidden", tower)
        self.assertIn("m->has_vision = 1", tower)
        self.assertEqual(tower.count("load_f32("), 25)
        self.assertNotIn("load_vision", tower)

    def test_payload_reads_and_release(self):
        with tempfile.TemporaryDirectory(prefix="glm53-text-loader-") as temporary:
            directory = Path(temporary)
            fixture(directory)
            binary = directory / ("probe.exe" if os.name == "nt" else "probe")
            command = shlex.split(os.environ.get("CC", "gcc")) + [
                "-O1", "-g", "-std=gnu11", "-pthread",
                str(ROOT / "tests/test_glm53_text_loader.c"),
                str(ROOT / "segment_runtime.c"), str(ROOT / "edge_runtime.c"),
                "-lm", "-o", str(binary)]
            command += shlex.split(os.environ.get("GLM53_LOADER_TEST_CFLAGS", ""))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            environment = dict(os.environ, GLM53_BITS="32")
            result = subprocess.run([str(binary), str(directory)], env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("after: Edge=0 Segment=0; normal=25: OK", result.stdout)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
