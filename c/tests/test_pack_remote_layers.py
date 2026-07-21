import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.pack_remote_layers import HEADER, RECORD, pack


class RemoteLayerPackTest(unittest.TestCase):
    def test_packs_complete_layer_index_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {}
            for eid in range(256):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    metadata[
                        f"model.layers.4.mlp.experts.{eid}.{projection}.weight"
                    ] = {"data_offsets": [0, 1]}
                    metadata[
                        f"model.layers.4.mlp.experts.{eid}.{projection}.weight.qs"
                    ] = {"data_offsets": [0, 1]}
            encoded = json.dumps(metadata, separators=(",", ":")).encode()
            shard = root / "out-00000.safetensors"
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x7f")
            output = root / "layer4.colirxp"

            count, size, _ = pack(root, output, [4])

            self.assertEqual(count, 256)
            self.assertEqual(size, output.stat().st_size)
            with output.open("rb") as file:
                header = HEADER.unpack(file.read(HEADER.size))
                self.assertEqual(header[4], 256)
                file.seek(header[5])
                first = RECORD.unpack(file.read(RECORD.size))
                self.assertEqual(first[:2], (4, 0))
                self.assertEqual(first[9:], (1,) * 6)
                file.seek(header[6])
                self.assertEqual(file.read(6), b"\x7f" * 6)


if __name__ == "__main__":
    unittest.main()
