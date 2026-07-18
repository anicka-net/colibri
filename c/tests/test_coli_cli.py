import os
import runpy
import stat
import tempfile
import unittest
from unittest.mock import patch


COLI = os.path.join(os.path.dirname(__file__), "..", "coli")


class StopMatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(COLI, run_name="coli_test")

    def test_matches_only_real_coli_serve_argv(self):
        match = self.module["is_serve_argv"]
        self.assertTrue(match([b"python3", b"/home/me/coli", b"serve", b"--port", b"8000"]))
        self.assertFalse(match([b"grep", b"-r", b"coli serve", b"."]))
        self.assertFalse(match([b"/home/me/coli", b"chat", b"--model", b"x"]))

    def test_pidfile_is_private_and_outside_shared_tmp(self):
        with tempfile.TemporaryDirectory() as runtime, patch.dict(
                os.environ, {"XDG_RUNTIME_DIR": runtime}):
            path = self.module["serve_pidfile"](8000)
            self.module["write_pidfile"](path, "/model")
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(os.path.dirname(path), runtime)


if __name__ == "__main__":
    unittest.main()
