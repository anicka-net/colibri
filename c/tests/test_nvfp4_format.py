import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tools"))
import nvfp4_format as nf
from convert_modelopt_nvfp4 import is_routed_expert, merge_expert_records


class Nvfp4FormatTest(unittest.TestCase):
    def test_every_e2m1_code_and_odd_input(self):
        codes = np.resize(np.arange(16, dtype=np.uint8), (3, 17))
        packed = np.zeros((3, 9), dtype=np.uint8)
        packed[:, :9] = codes[:, 0::2]
        packed[:, :8] |= codes[:, 1::2] << 4
        got = nf.unpack_e2m1(packed, 17)
        np.testing.assert_array_equal(got, nf.E2M1[codes])

    def test_e4m3fn_known_values_and_nan(self):
        raw = np.asarray([0x00, 0x01, 0x38, 0x40, 0x7E, 0xB8], dtype=np.uint8)
        np.testing.assert_array_equal(
            nf.decode_e4m3fn(raw),
            np.asarray([0.0, 2**-9, 1.0, 2.0, 448.0, -1.0], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "NaN"):
            nf.decode_e4m3fn(np.asarray([0x7F], dtype=np.uint8))

    def test_dequantize_uses_block_and_tensor_scale(self):
        codes = np.arange(17, dtype=np.uint8) % 16
        packed = np.zeros((1, 9), dtype=np.uint8)
        packed[0, :] = codes[0::2]
        packed[0, :8] |= codes[1::2] << 4
        scales = np.asarray([[0x38, 0x40]], dtype=np.uint8)  # 1, 2
        got = nf.dequantize_modelopt(packed, scales, 0.25, 17)
        expect = nf.E2M1[codes] * np.asarray([1.0] * 16 + [2.0], dtype=np.float32) * 0.25
        np.testing.assert_array_equal(got[0], expect)

    def test_cutlass_swizzle_round_trip_with_padding(self):
        src = np.arange(131 * 5, dtype=np.uint16).reshape(131, 5).astype(np.uint8)
        swizzled = nf.swizzle_scales_for_cutlass(src, 131, 65)
        self.assertEqual(swizzled.size, 2 * 2 * 512)
        np.testing.assert_array_equal(nf.unswizzle_scales_from_cutlass(swizzled, 131, 65), src)

    def test_manifest_round_trip_and_corruption(self):
        doc = nf.make_manifest("org/model", "a" * 40, nf.FORMAT_BF16)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / nf.MANIFEST_NAME
            path.write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(nf.load_manifest(td)["source"]["revision"], "a" * 40)
            bad = dict(doc)
            bad["routed_experts"] = dict(doc["routed_experts"], group_size=32)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(nf.ManifestError, "group_size"):
                nf.load_manifest(td)

    def test_aligned_safetensors_records(self):
        records = [
            [("expert.0.weight", nf.tensor_bytes(np.arange(17, dtype=np.uint8))),
             ("expert.0.scale", nf.tensor_bytes(np.asarray([1.0], dtype=np.float32)))],
            [("expert.1.weight", nf.tensor_bytes(np.arange(33, dtype=np.uint8))),
             ("expert.1.scale", nf.tensor_bytes(np.asarray([2.0], dtype=np.float32)))],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "experts.safetensors"
            offsets = nf.write_aligned_safetensors(path, records)
            data_start, header = nf.read_safetensors_header(path)
            self.assertEqual(data_start % 4096, 0)
            self.assertTrue(all(offset % 4096 == 0 for offset in offsets.values()))
            self.assertEqual(header["expert.1.weight"]["data_offsets"][1] -
                             header["expert.1.weight"]["data_offsets"][0], 33)

    def test_rejects_nonpositive_scales_and_wrong_shapes(self):
        packed = np.zeros((1, 8), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "positive"):
            nf.dequantize_modelopt(packed, np.zeros((1, 1), dtype=np.uint8), 0.5, 16)
        with self.assertRaisesRegex(ValueError, "shape"):
            nf.swizzle_scales_for_cutlass(np.ones((1, 2), dtype=np.uint8), 1, 16)

    def test_expert_records_can_span_source_shards(self):
        key = (12, 127)
        def record(projection):
            prefix = f"model.layers.12.mlp.experts.127.{projection}.weight"
            return [(prefix, nf.tensor_bytes(np.asarray([1], dtype=np.uint8))),
                    (prefix + ".nvfp4_scale", nf.tensor_bytes(np.asarray([0x38], dtype=np.uint8)))]
        pending = {}
        self.assertEqual(merge_expert_records(pending, {key: record("gate_proj")}), [])
        self.assertIn(key, pending)
        done = merge_expert_records(pending, {key: record("up_proj") + record("down_proj")})
        self.assertEqual(len(done), 1)
        self.assertNotIn(key, pending)

    def test_mtp_layer_is_not_a_routed_expert(self):
        base = "model.layers.{}.mlp.experts.0.down_proj.weight"
        self.assertIsNotNone(is_routed_expert(base.format(77), 78))
        self.assertIsNone(is_routed_expert(base.format(78), 78))


if __name__ == "__main__":
    unittest.main()
