"""Keep the run archive in S3: every file of every run is uploaded, then the heavy ones (the wavs
and the multi-MB run_meta.json / db_state_initial.json, ~99% of the bytes) are dropped locally
and downloaded again the first time something asks for them.

Bucket (VIEWER_S3_URI=s3://<bucket>/<prefix>):
    <prefix>/<run>/<file>             every file of the run, verbatim
    <prefix>/.manifests/<run>.json    what was uploaded, so any machine can pull the light copy
Data dir:
    <run>/<file>                      light files always; heavy files until offloaded / while cached
    .s3/<run>.json                    the run's manifest: {"files": {rel: {size, mtime, ...}}}
    .s3/cache.json                    heavy files the viewer fetched, evicted oldest first
    .s3/pending.json                  runs waiting for the background upload (survives a restart)

A manifest entry is written only after its upload succeeded, and a local file is deleted only
while it still matches its entry (size + mtime) and S3 lists an object of that size, so the
manifest is the record of what S3 holds. Wav entries carry the header (rate, frames) and
run_meta.json entries its small fields (everything but db_state): all the viewer reads from them.

The viewer uploads runs that arrive (Sync, drag & drop, Run test) by itself, and a run deleted in
the viewer is deleted from the bucket too. The CLI is for the first migration and maintenance:
    uv run python -m app.s3store status
    uv run python -m app.s3store push [RUN ...]          # all local runs by default
    uv run python -m app.s3store offload [RUN ...] [--yes]
    uv run python -m app.s3store pull [RUN ...] [--full] # all runs missing locally by default
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HEAVY_NAMES = frozenset({"run_meta.json", "db_state_initial.json"})
MANIFEST_DIR = ".manifests"
TRANSFER_WORKERS = 16
_SUMMARY_VALUE_LIMIT = 4096  # run_meta fields kept in the manifest (db_state is the multi-MB one)
_AUTH_MARKERS = ("ExpiredToken", "InvalidToken", "UnauthorizedSSOToken", "SSOTokenLoad", "TokenRetrieval",
                 "NoCredentials", "InvalidAccessKeyId", "SignatureDoesNotMatch")


def is_heavy(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(".wav") or name in HEAVY_NAMES


def _safe_rel(rel: str) -> bool:
    """A run-relative path the viewer would show: no absolute paths, '..' or dotfiles."""
    return bool(rel) and all(part and not part.startswith(".") for part in rel.split("/"))


def _matches(st: os.stat_result, entry: dict[str, Any] | None) -> bool:
    return bool(entry) and st.st_size == entry["size"] and int(st.st_mtime) == entry["mtime"]


def _local_files(run_dir: Path) -> dict[str, os.stat_result]:
    out = {}
    for p in run_dir.rglob("*"):
        rel = p.relative_to(run_dir).as_posix()
        if _safe_rel(rel) and p.is_file():
            out[rel] = p.stat()
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class S3Error(Exception):
    """An S3 call failed; the message says what to do (e.g. renew the AWS SSO login)."""


class S3Archive:
    def __init__(self, data_dir: Path, uri: str | None, profile: str | None = None, region: str | None = None,
                 storage_class: str = "INTELLIGENT_TIERING", cache_bytes: int = 5 * 2**30,
                 auto_offload: bool = True):
        if uri and (not uri.startswith("s3://") or not uri[5:].split("/", 1)[0]):
            raise ValueError(f"VIEWER_S3_URI must look like s3://<bucket>/<prefix>, got {uri!r}")
        self.data_dir = data_dir
        self.uri = uri
        self.bucket, _, prefix = (uri or "")[5:].partition("/")
        self.prefix = prefix.strip("/")
        self.profile, self.region, self.storage_class = profile, region, storage_class
        self.cache_bytes, self.auto_offload = cache_bytes, auto_offload
        self.local_dir = data_dir / ".s3"
        self._client_obj = None
        self._client_lock = threading.Lock()
        self._manifests: dict[str, tuple[int, dict[str, Any]]] = {}  # run -> (file mtime_ns, manifest)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._cache_lock = threading.Lock()
        # background uploads of runs that just arrived
        self._queue: list[str] = []
        self._queue_cv = threading.Condition()
        self._worker: threading.Thread | None = None
        self.pushing: str | None = None
        self.failed: dict[str, str] = {}
        self.error: str | None = None  # the last credentials / access problem, cleared by the next success

    @classmethod
    def from_env(cls, data_dir: Path) -> S3Archive:
        env = os.environ.get
        return cls(
            data_dir, (env("VIEWER_S3_URI") or "").strip() or None,
            profile=env("VIEWER_S3_PROFILE") or None,
            region=env("VIEWER_S3_REGION") or None,
            storage_class=env("VIEWER_S3_STORAGE_CLASS") or "INTELLIGENT_TIERING",
            cache_bytes=int(float(env("VIEWER_S3_CACHE_GB") or 5) * 2**30),
            auto_offload=(env("VIEWER_S3_AUTO_OFFLOAD") or "1").lower() not in ("0", "false", "no"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    # ---- S3 calls ----

    def _client(self):
        if not self.enabled:
            raise S3Error("S3 is not configured (set VIEWER_S3_URI in .env)")
        with self._client_lock:
            if self._client_obj is None:
                import boto3
                from botocore.config import Config

                try:
                    session = boto3.Session(profile_name=self.profile, region_name=self.region)
                    self._client_obj = session.client("s3", config=Config(
                        retries={"max_attempts": 5, "mode": "standard"},
                        max_pool_connections=TRANSFER_WORKERS * 4,  # parallel files x multipart threads
                    ))
                except Exception as e:
                    raise self._wrap(e) from e
            return self._client_obj

    def _wrap(self, e: Exception) -> S3Error:
        if isinstance(e, S3Error):
            return e
        text = f"{type(e).__name__}: {e}"
        if "ProfileNotFound" in text:
            self.error = f"AWS profile {self.profile!r} is not in ~/.aws/config (VIEWER_S3_PROFILE)"
            return S3Error(self.error)
        if any(m in text for m in _AUTH_MARKERS):
            login = "aws sso login" + (f" --profile {self.profile}" if self.profile else "")
            self.error = f"AWS credentials expired or missing; run `{login}` ({text})"
            with self._client_lock:
                self._client_obj = None  # pick up the new login on the next call
            return S3Error(self.error)
        if "AccessDenied" in text or "NoSuchBucket" in text or "AllAccessDisabled" in text:
            self.error = f"S3 refused access to {self.uri}: {text}"
            return S3Error(self.error)
        return S3Error(f"S3 error: {text}")

    def _call(self, fn: Callable, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            raise self._wrap(e) from e
        self.error = None
        return result

    def _key(self, run: str, rel: str = "") -> str:
        return "/".join(p for p in (self.prefix, run, rel) if p)

    def _manifest_key(self, run: str = "") -> str:
        return "/".join(p for p in (self.prefix, MANIFEST_DIR, f"{run}.json" if run else "") if p)

    def remote_runs(self) -> list[str]:
        """Runs whose manifest is in the bucket."""
        prefix = self._manifest_key() + "/"
        pages = self._call(lambda: list(self._client().get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix)))
        return sorted(o["Key"][len(prefix):-5] for page in pages for o in page.get("Contents", [])
                      if o["Key"].endswith(".json"))

    def _remote_sizes(self, run: str) -> dict[str, int]:
        prefix = self._key(run) + "/"
        pages = self._call(lambda: list(self._client().get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix)))
        return {o["Key"][len(prefix):]: o["Size"] for page in pages for o in page.get("Contents", [])}

    # ---- manifests ----

    def manifest(self, run: str) -> dict[str, Any] | None:
        """The run's local manifest (cached; callers must not modify it)."""
        path = self.local_dir / f"{run}.json"
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return None
        cached = self._manifests.get(run)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            return None
        self._manifests[run] = (mtime, data)
        return data

    def _write_manifest(self, run: str, manifest: dict[str, Any]) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        path = self.local_dir / f"{run}.json"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
        tmp.replace(path)

    def _split(self, path: Path) -> tuple[str, str] | None:
        """(run, run-relative path) of a file in the data dir."""
        try:
            parts = path.relative_to(self.data_dir).parts
        except ValueError:
            return None
        return (parts[0], "/".join(parts[1:])) if len(parts) >= 2 else None

    def entry(self, path: Path) -> dict[str, Any] | None:
        """The manifest entry of an archived file (None when it was never uploaded)."""
        split = self._split(path)
        m = self.manifest(split[0]) if split else None
        return m["files"].get(split[1]) if m else None

    def exists(self, path: Path) -> bool:
        """Whether the archive has this file, here or only in S3."""
        return path.exists() or self.entry(path) is not None

    def offloaded(self, directory: Path) -> dict[str, dict[str, Any]]:
        """Files under a run or scenario dir that are kept only in S3, keyed relative to it."""
        try:
            parts = directory.relative_to(self.data_dir).parts
        except ValueError:
            return {}
        m = self.manifest(parts[0]) if parts else None
        if not m:
            return {}
        prefix = "/".join(parts[1:]) + "/" if len(parts) > 1 else ""
        return {
            rel[len(prefix):]: e for rel, e in m["files"].items()
            if rel.startswith(prefix) and not (directory / rel[len(prefix):]).exists()
        }

    # ---- download on demand ----

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _download(self, run: str, rel: str, entry: dict[str, Any]) -> None:
        path = self.data_dir / run / rel
        tmp_dir = self.local_dir / "tmp"  # same filesystem, outside the run so Sync never sees a partial file
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{uuid.uuid4().hex}.part"
        try:
            self._call(self._client().download_file, self.bucket, self._key(run, rel), str(tmp))
            size = tmp.stat().st_size
            if size != entry["size"]:
                raise S3Error(f"S3 copy of {run}/{rel} has {size} bytes, the manifest says {entry['size']}")
            os.utime(tmp, (entry["mtime"], entry["mtime"]))  # keeps the Sync signature stable
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def fetch(self, path: Path) -> bool:
        """Download an archived file kept only in S3 back into place (it then counts toward the
        cache and may be evicted again). False when the archive has no such file."""
        if path.exists():
            return True
        split, entry = self._split(path), self.entry(path)
        if not split or entry is None or not _safe_rel(split[1]):
            return False
        with self._lock_for(f"fetch:{path}"):
            if path.exists():
                return True
            self._download(*split, entry)
        self._remember(f"{split[0]}/{split[1]}")
        return True

    def _remember(self, key: str) -> None:
        """Record a fetched file and evict the oldest fetched ones beyond the cache size."""
        cache_file = self.local_dir / "cache.json"
        with self._cache_lock:
            try:
                cache = json.loads(cache_file.read_text())
            except (OSError, ValueError):
                cache = {}
            cache[key] = time.time()
            sizes = {}
            for k in list(cache):
                try:
                    sizes[k] = (self.data_dir / k).stat().st_size
                except OSError:
                    del cache[k]  # evicted, offloaded or replaced since
            total = sum(sizes.values())
            for k in sorted(cache, key=cache.get):
                if total <= self.cache_bytes:
                    break
                if k != key and self._drop(self.data_dir / k):
                    total -= sizes[k]
                    del cache[k]
            tmp = cache_file.with_name(f"cache.json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(cache_file)

    def _drop(self, path: Path) -> bool:
        """Delete a local copy that is still the file S3 holds."""
        try:
            if _matches(path.stat(), self.entry(path)):
                path.unlink()
                return True
        except OSError:
            pass
        return False

    # ---- upload / offload / pull ----

    def _upload(self, run_dir: Path, rel: str) -> tuple[str, dict[str, Any]]:
        path = run_dir / rel
        st = path.stat()  # before the upload: if the file changes meanwhile, the next push sees it
        entry: dict[str, Any] = {"size": st.st_size, "mtime": int(st.st_mtime)}
        if path.name.endswith(".wav"):
            try:
                with wave.open(str(path)) as w:
                    entry.update(rate=w.getframerate(), frames=w.getnframes())
            except (OSError, EOFError, wave.Error):
                pass
        elif path.name == "run_meta.json":
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                meta = None
            if isinstance(meta, dict):
                entry["summary"] = {k: v for k, v in meta.items()
                                    if k != "db_state" and len(json.dumps(v)) < _SUMMARY_VALUE_LIMIT}
        self._call(self._client().upload_file, str(path), self.bucket, self._key(run_dir.name, rel),
                   ExtraArgs={"StorageClass": self.storage_class})
        return rel, entry

    def push_run(self, run: str, progress: Callable[[int, int], None] | None = None) -> tuple[int, int]:
        """Upload the run's new and changed files, then its manifest. Returns (files, bytes) uploaded."""
        run_dir = self.data_dir / run
        if not run_dir.is_dir():
            raise S3Error(f"no run {run!r} in {self.data_dir}")
        self._client()
        with self._lock_for(f"run:{run}"):
            if not run_dir.is_dir():  # deleted while this push waited for the lock
                return 0, 0
            files = dict((self.manifest(run) or {}).get("files") or {})
            local = _local_files(run_dir)
            todo = [rel for rel, st in local.items() if not _matches(st, files.get(rel))]
            sent = sent_bytes = 0
            last_save = time.monotonic()
            pool = ThreadPoolExecutor(TRANSFER_WORKERS)
            try:
                for fut in as_completed([pool.submit(self._upload, run_dir, rel) for rel in todo]):
                    rel, entry = fut.result()
                    files[rel] = entry
                    sent, sent_bytes = sent + 1, sent_bytes + entry["size"]
                    if progress:
                        progress(sent, len(todo))
                    if time.monotonic() - last_save > 30:  # an interrupted push resumes from here
                        self._write_manifest(run, {"run": run, "files": files})
                        last_save = time.monotonic()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
                if sent:
                    self._write_manifest(run, {"run": run, "files": files})
            # a light file that is gone locally left the run (a re-synced run lost it); a missing
            # heavy file is one that was offloaded
            files = {rel: e for rel, e in files.items() if rel in local or is_heavy(rel)}
            manifest = {"run": run, "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "files": files}
            body = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
            self._call(self._client().put_object, Bucket=self.bucket, Key=self._manifest_key(run), Body=body,
                       ContentType="application/json")
            self._write_manifest(run, manifest)
        return sent, sent_bytes

    def offload_run(self, run: str) -> tuple[int, int, int]:
        """Delete the run's heavy files locally where S3 holds the same file.
        Returns (files deleted, bytes freed, heavy files kept because S3 does not hold them yet)."""
        m = self.manifest(run)
        local = _local_files(self.data_dir / run)
        heavy = {rel: st for rel, st in local.items() if is_heavy(rel)}
        if not heavy:
            return 0, 0, 0
        if not m:
            return 0, 0, len(heavy)
        with self._lock_for(f"run:{run}"):
            remote = self._remote_sizes(run)
            deleted = freed = kept = 0
            for rel, st in heavy.items():
                entry = m["files"].get(rel)
                if _matches(st, entry) and remote.get(rel) == entry["size"]:
                    (self.data_dir / run / rel).unlink(missing_ok=True)
                    deleted, freed = deleted + 1, freed + st.st_size
                else:
                    kept += 1
        return deleted, freed, kept

    def pull_run(self, run: str, full: bool = False) -> tuple[int, int]:
        """Fetch a run's manifest from S3 and download its light files (every file with full=True)
        that are missing here. Files pulled this way are not evicted. Returns (files, bytes)."""
        body = self._call(lambda: self._client().get_object(
            Bucket=self.bucket, Key=self._manifest_key(run))["Body"].read())
        try:
            m = json.loads(body)
        except ValueError:
            m = None
        files = m.get("files") if isinstance(m, dict) else None
        if not isinstance(files, dict):
            raise S3Error(f"the manifest of {run} in S3 is not valid")
        with self._lock_for(f"run:{run}"):
            self._write_manifest(run, m)
            todo = [(rel, e) for rel, e in files.items()
                    if _safe_rel(rel) and (full or not is_heavy(rel)) and not (self.data_dir / run / rel).exists()]
            with ThreadPoolExecutor(TRANSFER_WORKERS) as pool:
                for fut in as_completed([pool.submit(self._download, run, rel, e) for rel, e in todo]):
                    fut.result()
        return len(todo), sum(e["size"] for _, e in todo)

    def delete_run(self, run: str, remove_local: Callable[[], None]) -> int:
        """Delete every object of the run and its manifest from the bucket, forget the run here, then
        call remove_local() while still holding the run's lock, so a queued upload cannot put it back.
        An S3 error leaves everything as it was. With S3 not configured only remove_local() runs.
        Returns the number of objects deleted (bucket versioning keeps them as noncurrent versions)."""
        if not self.enabled:
            remove_local()
            return 0
        with self._lock_for(f"run:{run}"):
            keys = [self._key(run, rel) for rel in self._remote_sizes(run)]
            batch = [*keys, self._manifest_key(run)]
            for i in range(0, len(batch), 1000):  # delete_objects takes at most 1000 keys
                resp = self._call(self._client().delete_objects, Bucket=self.bucket, Delete={
                    "Objects": [{"Key": k} for k in batch[i:i + 1000]], "Quiet": True})
                if resp.get("Errors"):
                    err = resp["Errors"][0]
                    raise S3Error(f"S3 did not delete {err.get('Key')}: {err.get('Code')} {err.get('Message')}")
            (self.local_dir / f"{run}.json").unlink(missing_ok=True)
            self._manifests.pop(run, None)
            remove_local()
            with self._queue_cv:
                if run in self._queue:
                    self._queue.remove(run)
                self.failed.pop(run, None)
                self._save_pending()
        return len(keys)

    # ---- background uploads in the viewer ----

    def start(self) -> None:
        """Resume the uploads a stopped viewer left queued."""
        if not self.enabled:
            return
        try:
            pending = json.loads((self.local_dir / "pending.json").read_text())
        except (OSError, ValueError):
            return
        if isinstance(pending, list):
            self.schedule_push(*[r for r in pending if isinstance(r, str)])

    def schedule_push(self, *runs: str) -> None:
        """Upload runs that just arrived, in the background, then (auto_offload) drop their heavy
        files. Uploads that failed earlier are retried too."""
        if not self.enabled:
            return
        with self._queue_cv:
            for run in (*runs, *self.failed):
                if run not in self._queue:  # a run being pushed is queued again: it may have new files
                    self._queue.append(run)
            self.failed.clear()
            self._save_pending()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="s3-push", daemon=True)
                self._worker.start()
            self._queue_cv.notify()

    def _save_pending(self) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        runs = [*([self.pushing] if self.pushing else []), *self._queue, *self.failed]
        tmp = self.local_dir / f"pending.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(list(dict.fromkeys(runs))))
        tmp.replace(self.local_dir / "pending.json")

    def _work(self) -> None:
        while True:
            with self._queue_cv:
                while not self._queue:
                    self._queue_cv.wait()
                self.pushing = self._queue.pop(0)
            run = self.pushing
            try:
                if (self.data_dir / run).is_dir():
                    self.push_run(run)
                    if self.auto_offload:
                        self.offload_run(run)
            except Exception as e:  # shown in the viewer's sidebar; the next push retries it
                self.failed[run] = str(e)
            with self._queue_cv:
                self.pushing = None
                self._save_pending()

    def status(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._queue_cv:
            queued = len(self._queue) + (1 if self.pushing else 0)
        return {"uri": self.uri, "queued": queued, "failed": dict(self.failed), "error": self.error}


# ---- CLI ----

def _cli_status(a: S3Archive) -> int:
    print(f"bucket:   {a.uri}  (profile {a.profile or 'default'}, storage class {a.storage_class})")
    print(f"data dir: {a.data_dir}")
    try:
        remote = a.remote_runs()
        print(f"S3:       reachable, {len(remote)} runs uploaded")
    except S3Error as e:
        print(f"S3:       {e}")
        remote = None
    runs = sorted(d.name for d in a.data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    unpushed = offloadable = offloadable_bytes = pending_bytes = 0
    for run in runs:
        files = (a.manifest(run) or {}).get("files") or {}
        stale = False
        for rel, st in _local_files(a.data_dir / run).items():
            if not _matches(st, files.get(rel)):
                stale, pending_bytes = True, pending_bytes + st.st_size
            elif is_heavy(rel):
                offloadable, offloadable_bytes = offloadable + 1, offloadable_bytes + st.st_size
        unpushed += stale
    print(f"local:    {len(runs)} runs, {unpushed} with files not uploaded yet ({_fmt_bytes(pending_bytes)})")
    print(f"          {offloadable} heavy files ({_fmt_bytes(offloadable_bytes)}) are in S3 and could be offloaded")
    if remote is not None:
        missing = sorted(set(remote) - set(runs))
        print(f"          {len(missing)} runs are only in S3 (pull fetches their light copy)")
    return 0


def _cli_push(a: S3Archive, runs: list[str]) -> int:
    runs = runs or sorted(d.name for d in a.data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    total_files = total_bytes = 0
    failures = []
    started = time.monotonic()
    tty = sys.stdout.isatty()
    cr = "\r\x1b[K" if tty else ""  # redraw the run's line in a terminal, plain lines in a log
    for i, run in enumerate(runs, 1):
        t0 = time.monotonic()

        def progress(done: int, todo: int, run=run, i=i) -> None:
            if done % 50 == 0 or done == todo:
                print(f"{cr}[{i}/{len(runs)}] {run}: {done}/{todo} files", end="", flush=True)

        try:
            files, size = a.push_run(run, progress if tty else None)
        except S3Error as e:
            print(f"{cr}[{i}/{len(runs)}] {run}: FAILED {e}")
            if a.error:  # credentials or access: every other run would fail the same way
                return 1
            failures.append(run)
            continue
        except KeyboardInterrupt:
            print("\ninterrupted; files uploaded so far are recorded, run push again to continue")
            return 130
        total_files, total_bytes = total_files + files, total_bytes + size
        rate = size / max(time.monotonic() - t0, 1e-6)
        print(f"{cr}[{i}/{len(runs)}] {run}: " + (
            f"uploaded {files} files, {_fmt_bytes(size)} ({_fmt_bytes(rate)}/s)" if files else "up to date"))
    print(f"done: {total_files} files, {_fmt_bytes(total_bytes)} in {time.monotonic() - started:.0f} s"
          + (f"; failed: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


def _cli_offload(a: S3Archive, runs: list[str], yes: bool) -> int:
    runs = runs or sorted(p.stem for p in a.local_dir.glob("*.json") if p.stem not in ("cache", "pending"))
    if not yes:
        answer = input(f"Delete the local heavy files of {len(runs)} runs where S3 holds the same file? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return 1
    total_files = total_bytes = total_kept = 0
    for run in runs:
        try:
            files, size, kept = a.offload_run(run)
        except S3Error as e:
            print(f"{run}: FAILED {e}")
            return 1
        total_files, total_bytes, total_kept = total_files + files, total_bytes + size, total_kept + kept
        if files or kept:
            print(f"{run}: freed {_fmt_bytes(size)} ({files} files)" + (f", kept {kept} not in S3 yet" if kept else ""))
    print(f"done: freed {_fmt_bytes(total_bytes)} ({total_files} files)"
          + (f"; {total_kept} heavy files kept because S3 does not hold them yet (push first)" if total_kept else ""))
    return 0


def _cli_pull(a: S3Archive, runs: list[str], full: bool) -> int:
    if not runs:
        local = {d.name for d in a.data_dir.iterdir() if d.is_dir()}
        runs = [r for r in a.remote_runs() if r not in local]
    for run in runs:
        files, size = a.pull_run(run, full=full)
        print(f"{run}: downloaded {files} files, {_fmt_bytes(size)}")
    print(f"done: {len(runs)} runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dotenv import load_dotenv

    base = Path(__file__).resolve().parent.parent
    load_dotenv(base / ".env")
    data_dir = Path(os.environ.get("VIEWER_DATA_DIR") or base / "data").expanduser()
    ap = argparse.ArgumentParser(prog="python -m app.s3store", description="Keep the run archive in S3.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="what is uploaded, what could be offloaded")
    p = sub.add_parser("push", help="upload new and changed files (all local runs by default)")
    p.add_argument("runs", nargs="*")
    p = sub.add_parser("offload", help="delete local heavy files that S3 holds (all uploaded runs by default)")
    p.add_argument("runs", nargs="*")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p = sub.add_parser("pull", help="download light copies of runs (all runs missing locally by default)")
    p.add_argument("runs", nargs="*")
    p.add_argument("--full", action="store_true", help="also download the heavy files (they are not evicted)")
    args = ap.parse_args(argv)
    archive = S3Archive.from_env(data_dir)
    if not archive.enabled:
        print("VIEWER_S3_URI is not set; add it to .env (see .env.example)", file=sys.stderr)
        return 2
    try:
        if args.cmd == "status":
            return _cli_status(archive)
        if args.cmd == "push":
            return _cli_push(archive, args.runs)
        if args.cmd == "offload":
            return _cli_offload(archive, args.runs, args.yes)
        return _cli_pull(archive, args.runs, args.full)
    except S3Error as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
