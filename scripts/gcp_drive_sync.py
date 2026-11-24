#!/usr/bin/env python3
"""GCP-authenticated Google Drive v3 sync strictly pinned to MyDrive/qwen3-asr.

Uses `gcloud auth print-access-token --account <account>` and pins the canonical
project root folder ID (1OWJDWy_iQ1N8hZbLKQOVWhYqwCyDIr-Z, name='qwen3-asr') so
no artifact can ever be uploaded to another project or outside qwen3-asr/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_ACCOUNT = "hawk.wu4@gmail.com"
CANONICAL_ROOT_FOLDER_ID = "1OWJDWy_iQ1N8hZbLKQOVWhYqwCyDIr-Z"
CANONICAL_ROOT_FOLDER_NAME = "qwen3-asr"
FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SYNC_SCHEMA = "qwen3-asr-gcp-drive-sync-v1"
CHUNK_BYTES = 16 * 1024 * 1024  # 16 MiB resumable upload chunks (multiple of 256 KiB)


def validate_remote_subpath(subpath: str) -> list[str]:
    """Validate and split a relative POSIX path under qwen3-asr/."""
    raw = subpath.strip("/")
    if not raw:
        return []
    if subpath.startswith("/") or "\\" in subpath:
        raise ValueError(f"remote subpath must be relative POSIX without leading slash: {subpath!r}")
    parts = [p for p in raw.split("/") if p]
    for p in parts:
        if p in (".", ".."):
            raise ValueError(f"remote subpath traversal rejected: {subpath!r}")
    return parts


def file_hashes(path: Path) -> tuple[int, str, str]:
    """Return (byte_size, md5_hex, sha256_hex) for a local file."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            total += len(chunk)
            md5.update(chunk)
            sha.update(chunk)
    return total, md5.hexdigest(), sha.hexdigest()


def _find_gcloud_binary() -> str:
    found = shutil.which("gcloud")
    if found:
        return found
    for candidate in (
        Path.home() / "Desktop" / "google-cloud-sdk" / "bin" / "gcloud",
        Path.home() / "google-cloud-sdk" / "bin" / "gcloud",
        Path("/opt/homebrew/bin/gcloud"),
        Path("/usr/local/bin/gcloud"),
        Path("/usr/bin/gcloud"),
    ):
        if candidate.is_file():
            return str(candidate)
    return "gcloud"


def get_gcloud_access_token(account: str = DEFAULT_ACCOUNT) -> str:
    env_tok = os.environ.get("GCP_DRIVE_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok
    cmd = [_find_gcloud_binary(), "auth", "print-access-token", "--account", account]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"gcloud token failed for {account}: {proc.stderr.strip()}")
    return proc.stdout.strip()


class GcpDriveClient:
    """Minimal, auditable Google Drive v3 client pinned to MyDrive/qwen3-asr."""

    def __init__(
        self,
        *,
        account: str = DEFAULT_ACCOUNT,
        root_folder_id: str = CANONICAL_ROOT_FOLDER_ID,
        expected_root_name: str = CANONICAL_ROOT_FOLDER_NAME,
        token_provider: Callable[[], str] | None = None,
        http_opener: Callable[..., Any] | None = None,
    ) -> None:
        self.account = account
        self.root_folder_id = root_folder_id
        self.expected_root_name = expected_root_name
        self._token_provider = token_provider or (lambda: get_gcloud_access_token(account))
        self._http_opener = http_opener or urllib.request.urlopen
        self._token: str | None = None
        self._token_ts: float = 0.0
        self._folder_cache: dict[tuple[str, str], str] = {}
        self._root_verified = False

    def _bearer(self, force_refresh: bool = False) -> str:
        now = time.monotonic()
        if force_refresh or self._token is None or (now - self._token_ts) > 2400.0:
            self._token = self._token_provider()
            self._token_ts = now
        return self._token

    def _json_request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        for attempt in range(retries):
            headers = {"Authorization": f"Bearer {self._bearer(force_refresh=(attempt > 0))}"}
            if body is not None:
                headers["Content-Type"] = "application/json; charset=utf-8"
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with self._http_opener(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 429, 500, 502, 503, 504) and attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise RuntimeError(f"Drive API {method} {url} -> HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError:
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"Drive API retries exhausted for {url}")

    def verify_root(self) -> dict[str, Any]:
        """Verify the pinned root folder ID actually refers to the project's qwen3-asr folder."""
        if self._root_verified:
            return {"id": self.root_folder_id, "name": self.expected_root_name}
        url = (
            f"https://www.googleapis.com/drive/v3/files/{urllib.parse.quote(self.root_folder_id)}"
            "?fields=id,name,mimeType,trashed"
        )
        meta = self._json_request(url)
        if meta.get("id") != self.root_folder_id:
            raise ValueError(f"Drive root ID mismatch: {meta.get('id')!r} != {self.root_folder_id!r}")
        if meta.get("name") != self.expected_root_name:
            raise ValueError(
                f"refusing sync: Drive folder {self.root_folder_id} is named {meta.get('name')!r}, "
                f"expected {self.expected_root_name!r}"
            )
        if meta.get("mimeType") != FOLDER_MIME or meta.get("trashed") is True:
            raise ValueError(f"Drive root {self.root_folder_id} is not an active folder: {meta}")
        self._root_verified = True
        return meta

    def find_child(self, parent_id: str, name: str, *, folder_only: bool = False) -> dict[str, Any] | None:
        safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
        q = f"'{parent_id}' in parents and name='{safe_name}' and trashed=false"
        if folder_only:
            q += f" and mimeType='{FOLDER_MIME}'"
        url = (
            "https://www.googleapis.com/drive/v3/files?"
            + urllib.parse.urlencode(
                {
                    "q": q,
                    "fields": "files(id,name,mimeType,size,md5Checksum,modifiedTime)",
                    "pageSize": "10",
                }
            )
        )
        files = self._json_request(url).get("files", [])
        if not files:
            return None
        return files[0]

    def ensure_folder(self, parts: list[str]) -> str:
        self.verify_root()
        current = self.root_folder_id
        for part in parts:
            cache_key = (current, part)
            if cache_key in self._folder_cache:
                current = self._folder_cache[cache_key]
                continue
            existing = self.find_child(current, part, folder_only=True)
            if existing is not None:
                fid = str(existing["id"])
            else:
                created = self._json_request(
                    "https://www.googleapis.com/drive/v3/files?fields=id,name,mimeType",
                    method="POST",
                    payload={"name": part, "mimeType": FOLDER_MIME, "parents": [current]},
                )
                fid = str(created["id"])
            self._folder_cache[cache_key] = fid
            current = fid
        return current

    def list_folder(self, subpath: str) -> list[dict[str, Any]]:
        parts = validate_remote_subpath(subpath)
        self.verify_root()
        current = self.root_folder_id
        for part in parts:
            existing = self.find_child(current, part, folder_only=True)
            if existing is None:
                return []
            current = str(existing["id"])
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "q": f"'{current}' in parents and trashed=false",
                "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum,modifiedTime)",
                "pageSize": "200",
            }
            if page_token:
                params["pageToken"] = page_token
            url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
            data = self._json_request(url)
            items.extend(data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return items

    def upload_file(self, local_path: Path, remote_rel: str) -> dict[str, Any]:
        """Upload (or update in place) one file under qwen3-asr/ and verify remote size + MD5."""
        if not local_path.is_file():
            raise FileNotFoundError(f"local file missing: {local_path}")
        parts = validate_remote_subpath(remote_rel)
        if not parts:
            raise ValueError("remote_rel must include a filename")
        parent_id = self.ensure_folder(parts[:-1])
        filename = parts[-1]
        local_size, local_md5, local_sha256 = file_hashes(local_path)

        existing = self.find_child(parent_id, filename, folder_only=False)
        if (
            existing is not None
            and existing.get("mimeType") != FOLDER_MIME
            and int(existing.get("size") or -1) == local_size
            and str(existing.get("md5Checksum") or "").lower() == local_md5
        ):
            return {
                "remote_path": "/".join(parts),
                "file_id": str(existing["id"]),
                "bytes": local_size,
                "md5": local_md5,
                "sha256": local_sha256,
                "action": "skipped_verified",
            }

        # Initiate resumable upload session
        if existing is not None and existing.get("mimeType") != FOLDER_MIME:
            file_id = str(existing["id"])
            init_url = (
                f"https://www.googleapis.com/upload/drive/v3/files/{urllib.parse.quote(file_id)}"
                "?uploadType=resumable&fields=id,name,size,md5Checksum"
            )
            init_method = "PATCH"
            metadata: dict[str, Any] = {"name": filename}
        else:
            init_url = (
                "https://www.googleapis.com/upload/drive/v3/files"
                "?uploadType=resumable&fields=id,name,size,md5Checksum"
            )
            init_method = "POST"
            metadata = {"name": filename, "parents": [parent_id]}

        init_body = json.dumps(metadata).encode("utf-8")
        init_req = urllib.request.Request(
            init_url,
            data=init_body,
            headers={
                "Authorization": f"Bearer {self._bearer()}",
                "Content-Type": "application/json; charset=utf-8",
                "X-Upload-Content-Type": "application/octet-stream",
                "X-Upload-Content-Length": str(local_size),
            },
            method=init_method,
        )
        with self._http_opener(init_req, timeout=60) as resp:
            session_uri = resp.headers.get("Location")
        if not session_uri:
            raise RuntimeError(f"Drive resumable upload did not return Location header for {remote_rel}")

        result_meta: dict[str, Any] = {}
        if local_size == 0:
            put_req = urllib.request.Request(
                session_uri,
                data=b"",
                headers={"Content-Length": "0", "Content-Type": "application/octet-stream"},
                method="PUT",
            )
            with self._http_opener(put_req, timeout=60) as resp:
                result_meta = json.loads(resp.read().decode("utf-8"))
        else:
            with local_path.open("rb") as handle:
                offset = 0
                while offset < local_size:
                    chunk = handle.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    end = offset + len(chunk) - 1
                    for attempt in range(4):
                        put_req = urllib.request.Request(
                            session_uri,
                            data=chunk,
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Content-Length": str(len(chunk)),
                                "Content-Range": f"bytes {offset}-{end}/{local_size}",
                            },
                            method="PUT",
                        )
                        try:
                            with self._http_opener(put_req, timeout=300) as resp:
                                body = resp.read().decode("utf-8")
                                if body:
                                    result_meta = json.loads(body)
                            break
                        except urllib.error.HTTPError as exc:
                            if exc.code == 308:
                                break
                            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                                time.sleep(2.0 * (attempt + 1))
                                continue
                            detail = exc.read().decode("utf-8", errors="replace")[:400]
                            raise RuntimeError(
                                f"Drive chunk upload failed for {remote_rel} at {offset}: HTTP {exc.code} {detail}"
                            ) from exc
                    offset = end + 1

        remote_size = int(result_meta.get("size") or -1)
        remote_md5 = str(result_meta.get("md5Checksum") or "").lower()
        if remote_size != local_size or remote_md5 != local_md5:
            raise RuntimeError(
                f"Drive upload verification failed for {remote_rel}: "
                f"local=({local_size}, {local_md5}) != remote=({remote_size}, {remote_md5})"
            )
        return {
            "remote_path": "/".join(parts),
            "file_id": str(result_meta["id"]),
            "bytes": local_size,
            "md5": local_md5,
            "sha256": local_sha256,
            "action": "updated" if existing is not None else "uploaded",
        }

    def upload_dir(
        self,
        local_dir: Path,
        remote_prefix: str,
        *,
        exclude_globs: tuple[str, ...] = ("*.tmp", ".DS_Store"),
    ) -> dict[str, Any]:
        """Recursively upload all files under local_dir to qwen3-asr/<remote_prefix>/ and verify MD5."""
        if not local_dir.is_dir():
            raise FileNotFoundError(f"local dir missing: {local_dir}")
        prefix_parts = validate_remote_subpath(remote_prefix)
        records: list[dict[str, Any]] = []
        for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            if any(path.match(pat) for pat in exclude_globs):
                continue
            rel = path.relative_to(local_dir).as_posix()
            remote_rel = "/".join([*prefix_parts, rel])
            rec = self.upload_file(path, remote_rel)
            records.append(rec)
        return {
            "schema_version": DRIVE_SYNC_SCHEMA,
            "synced_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "account": self.account,
            "root_folder_id": self.root_folder_id,
            "root_folder_name": self.expected_root_name,
            "remote_prefix": "/".join(prefix_parts),
            "file_count": len(records),
            "total_bytes": sum(int(r["bytes"]) for r in records),
            "files": records,
        }

    def download_file(self, remote_rel: str, local_dest: Path) -> dict[str, Any]:
        """Download a file from qwen3-asr/<remote_rel> and verify local size + MD5 against Drive metadata."""
        parts = validate_remote_subpath(remote_rel)
        if not parts:
            raise ValueError("remote_rel must include a filename")
        self.verify_root()
        current = self.root_folder_id
        for folder in parts[:-1]:
            found = self.find_child(current, folder, folder_only=True)
            if found is None:
                raise FileNotFoundError(f"Drive folder not found: {folder} in {remote_rel}")
            current = str(found["id"])
        meta = self.find_child(current, parts[-1], folder_only=False)
        if meta is None or meta.get("mimeType") == FOLDER_MIME:
            raise FileNotFoundError(f"Drive file not found: {remote_rel}")
        file_id = str(meta["id"])
        expected_size = int(meta.get("size") or -1)
        expected_md5 = str(meta.get("md5Checksum") or "").lower()
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_dest.with_name(local_dest.name + ".tmp")
        url = f"https://www.googleapis.com/drive/v3/files/{urllib.parse.quote(file_id)}?alt=media"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._bearer()}"})
        with self._http_opener(req, timeout=300) as resp, tmp.open("wb") as out:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
        size, md5, sha = file_hashes(tmp)
        if size != expected_size or (expected_md5 and md5 != expected_md5):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Drive download checksum mismatch for {remote_rel}: {(size, md5)} != {(expected_size, expected_md5)}")
        os.replace(tmp, local_dest)
        return {"remote_path": "/".join(parts), "file_id": file_id, "bytes": size, "md5": md5, "sha256": sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GCP 授权的 Google Drive 同步工具（严格锁定 MyDrive/qwen3-asr）")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--root-folder-id", default=CANONICAL_ROOT_FOLDER_ID)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ls = sub.add_parser("ls", help="列出 qwen3-asr 下指定子目录")
    p_ls.add_argument("--remote-subpath", default="")

    p_uf = sub.add_parser("upload-file", help="上传单个文件并做远端 MD5/大小校验")
    p_uf.add_argument("--local-path", required=True, type=Path)
    p_uf.add_argument("--remote-path", required=True)

    p_ud = sub.add_parser("upload-dir", help="递归上传目录并写出同步清单")
    p_ud.add_argument("--local-dir", required=True, type=Path)
    p_ud.add_argument("--remote-prefix", required=True)
    p_ud.add_argument("--manifest-out", type=Path, default=None)

    p_df = sub.add_parser("download-file", help="下载单个文件并核对 MD5")
    p_df.add_argument("--remote-path", required=True)
    p_df.add_argument("--local-path", required=True, type=Path)

    args = parser.parse_args(argv)
    client = GcpDriveClient(account=args.account, root_folder_id=args.root_folder_id)
    try:
        if args.command == "ls":
            items = client.list_folder(args.remote_subpath)
            print(json.dumps({"root_folder_id": args.root_folder_id, "items": items}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "upload-file":
            rec = client.upload_file(args.local_path, args.remote_path)
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            return 0
        if args.command == "upload-dir":
            summary = client.upload_dir(args.local_dir, args.remote_prefix)
            if args.manifest_out is not None:
                args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
                args.manifest_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "remote_prefix": summary["remote_prefix"],
                        "file_count": summary["file_count"],
                        "total_bytes": summary["total_bytes"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "download-file":
            rec = client.download_file(args.remote_path, args.local_path)
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"GCP_DRIVE_SYNC_ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
