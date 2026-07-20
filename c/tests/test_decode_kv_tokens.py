import os
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


ROOT = Path(__file__).resolve().parents[1]


class DecodeKvTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "decode-kv-tokens"
        cc = shlex.split(os.environ.get("CC", "cc"))
        subprocess.run(
            cc + ["-O0", str(ROOT / "tools/decode_kv_tokens.c"),
                  "-lm", "-o", str(cls.binary)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def write_checkpoint(path, count, size=None):
        header = [0] * 8
        header[6] = count
        with path.open("wb") as f:
            f.write(struct.pack("<8s8i", b"COLIKT1\0", *header))
            if size is not None:
                f.truncate(size)

    def test_rejects_truncated_checkpoint_before_allocating(self):
        path = Path(self.tmp.name) / "short.tok"
        self.write_checkpoint(path, 5)
        run = subprocess.run(
            [self.binary, "unused-tokenizer", path, "0", "5"],
            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(run.returncode, 1)
        self.assertIn("short token checkpoint", run.stderr)

    def test_rejects_empty_range_argument(self):
        path = Path(self.tmp.name) / "range.tok"
        self.write_checkpoint(path, 0, 40)
        run = subprocess.run(
            [self.binary, "unused-tokenizer", path, "", "0"],
            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(run.returncode, 1)
        self.assertIn("invalid token range", run.stderr)

    @unittest.skipUnless(
        resource and hasattr(resource, "RLIMIT_AS") and sys.platform != "darwin",
        "requires a usable RLIMIT_AS",
    )
    def test_reports_id_allocation_failure(self):
        count = 8_000_000
        path = Path(self.tmp.name) / "large.tok"
        self.write_checkpoint(path, count, 40 + count * 4)

        def limit_address_space():
            limit = 24 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        run = subprocess.run(
            [self.binary, "unused-tokenizer", path, "0", str(count)],
            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=limit_address_space,
        )
        self.assertEqual(run.returncode, 1)
        self.assertIn("out of memory reading", run.stderr)


if __name__ == "__main__":
    unittest.main()
