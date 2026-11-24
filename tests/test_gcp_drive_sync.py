from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.gcp_drive_sync import (
    CANONICAL_ROOT_FOLDER_ID,
    FOLDER_MIME,
    GcpDriveClient,
    get_gcloud_access_token,
    validate_remote_subpath,
)


class _FakeResp:
    def __init__(self, body: bytes = b"{}", headers: dict[str, str] | None = None) -> None:
        self._buf = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class GcpDriveSyncTest(unittest.TestCase):
    def test_validate_remote_subpath_rejects_traversal_and_absolute(self) -> None:
        self.assertEqual(validate_remote_subpath("data/slr28/v1"), ["data", "slr28", "v1"])
        self.assertEqual(validate_remote_subpath(""), [])
        for bad in ("/etc/passwd", "../other-project/file", "runs/../escape", "a\\b"):
            with self.assertRaises(ValueError):
                validate_remote_subpath(bad)

    def test_env_token_override_works_without_gcloud_subprocess(self) -> None:
        with mock.patch.dict(os.environ, {"GCP_DRIVE_ACCESS_TOKEN": "ya29.test-ephemeral"}):
            self.assertEqual(get_gcloud_access_token(), "ya29.test-ephemeral")

    def test_verify_root_rejects_wrong_folder_name(self) -> None:
        def opener(req: Any, timeout: int = 60) -> _FakeResp:
            del req, timeout
            return _FakeResp(
                json.dumps({"id": CANONICAL_ROOT_FOLDER_ID, "name": "other-project", "mimeType": FOLDER_MIME, "trashed": False}).encode()
            )

        client = GcpDriveClient(token_provider=lambda: "fake-tok", http_opener=opener)
        with self.assertRaises(ValueError):
            client.verify_root()

    def test_upload_file_verifies_remote_size_and_md5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "hello.txt"
            payload = b"qwen3-asr-track3-payload"
            sample.write_bytes(payload)
            expected_md5 = hashlib.md5(payload).hexdigest()

            def opener(req: Any, timeout: int = 60) -> _FakeResp:
                del timeout
                url = req.full_url
                method = req.get_method()
                if f"/files/{CANONICAL_ROOT_FOLDER_ID}?fields=" in url:
                    return _FakeResp(
                        json.dumps({"id": CANONICAL_ROOT_FOLDER_ID, "name": "qwen3-asr", "mimeType": FOLDER_MIME, "trashed": False}).encode()
                    )
                if method == "GET" and "/files?q=" in url:
                    return _FakeResp(json.dumps({"files": []}).encode())
                if method == "POST" and "uploadType=resumable" not in url:
                    return _FakeResp(json.dumps({"id": "folder-data", "name": "data", "mimeType": FOLDER_MIME}).encode())
                if method == "POST" and "uploadType=resumable" in url:
                    return _FakeResp(b"", headers={"Location": "https://upload.fake/session1"})
                if method == "PUT" and url == "https://upload.fake/session1":
                    return _FakeResp(
                        json.dumps({"id": "file-123", "name": "hello.txt", "size": str(len(payload)), "md5Checksum": expected_md5}).encode()
                    )
                raise AssertionError(f"unexpected request {method} {url}")

            client = GcpDriveClient(token_provider=lambda: "fake-tok", http_opener=opener)
            rec = client.upload_file(sample, "data/hello.txt")
            self.assertEqual(rec["action"], "uploaded")
            self.assertEqual(rec["md5"], expected_md5)
            self.assertEqual(rec["bytes"], len(payload))

    def test_upload_file_fails_on_remote_md5_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "hello.txt"
            sample.write_bytes(b"abc")

            def opener(req: Any, timeout: int = 60) -> _FakeResp:
                del timeout
                url = req.full_url
                method = req.get_method()
                if f"/files/{CANONICAL_ROOT_FOLDER_ID}?fields=" in url:
                    return _FakeResp(
                        json.dumps({"id": CANONICAL_ROOT_FOLDER_ID, "name": "qwen3-asr", "mimeType": FOLDER_MIME, "trashed": False}).encode()
                    )
                if method == "GET" and "/files?q=" in url:
                    return _FakeResp(json.dumps({"files": []}).encode())
                if method == "POST" and "uploadType=resumable" in url:
                    return _FakeResp(b"", headers={"Location": "https://upload.fake/session2"})
                if method == "PUT":
                    return _FakeResp(
                        json.dumps({"id": "file-bad", "name": "hello.txt", "size": "3", "md5Checksum": "0" * 32}).encode()
                    )
                raise AssertionError(f"unexpected {method} {url}")

            client = GcpDriveClient(token_provider=lambda: "fake-tok", http_opener=opener)
            with self.assertRaises(RuntimeError):
                client.upload_file(sample, "hello.txt")


if __name__ == "__main__":
    unittest.main()
