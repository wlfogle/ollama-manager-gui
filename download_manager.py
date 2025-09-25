import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Optional, Callable, List, Dict
import requests

class DownloadStatus:
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"
    IMPORTING = "importing"

class DownloadJob:
    def __init__(self, *,
                 job_id: str,
                 url: str,
                 dest: Path,
                 provider: str,
                 suggested_name: Optional[str] = None,
                 repo_id: Optional[str] = None,
                 gguf_path: Optional[str] = None,
                 size: Optional[int] = None,
                 sha256: Optional[str] = None,
                 bandwidth_limit_bps: Optional[int] = None,
                 ollama_import: Optional[Callable[[str, Path], None]] = None,
                 ): 
        self.job_id = job_id
        self.url = url
        self.dest = dest
        self.provider = provider
        self.suggested_name = suggested_name
        self.repo_id = repo_id
        self.gguf_path = gguf_path
        self.size = size
        self.sha256 = (sha256 or '').lower() or None
        self.bandwidth_limit_bps = bandwidth_limit_bps
        self.ollama_import = ollama_import

        self.status = DownloadStatus.QUEUED
        self.error: Optional[str] = None
        self.downloaded: int = 0
        self.total: Optional[int] = size
        self.start_ts: Optional[float] = None
        self.last_update_ts: float = time.time()
        self._pause_ev = threading.Event()
        self._pause_ev.set()  # initially not paused
        self._cancel = False
        self._lock = threading.Lock()

    def pause(self):
        with self._lock:
            if self.status in (DownloadStatus.RUNNING, DownloadStatus.QUEUED):
                self.status = DownloadStatus.PAUSED
                self._pause_ev.clear()

    def resume(self):
        with self._lock:
            if self.status == DownloadStatus.PAUSED:
                self.status = DownloadStatus.QUEUED
                self._pause_ev.set()

    def cancel(self):
        with self._lock:
            self._cancel = True
            self._pause_ev.set()

class DownloadManager:
    def __init__(self, *,
                 max_concurrency: int = 2,
                 bandwidth_limit_bps: Optional[int] = None,
                 on_progress: Optional[Callable[[DownloadJob], None]] = None,
                 on_done: Optional[Callable[[DownloadJob], None]] = None,
                 on_import: Optional[Callable[[DownloadJob], None]] = None,
                 ):
        self.max_concurrency = max(1, int(max_concurrency))
        self.bandwidth_limit_bps = bandwidth_limit_bps
        self.on_progress = on_progress or (lambda job: None)
        self.on_done = on_done or (lambda job: None)
        self.on_import = on_import or (lambda job: None)

        self._jobs: Dict[str, DownloadJob] = {}
        self._queue: List[str] = []
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._stop = False
        self._ensure_workers()

    def set_max_concurrency(self, n: int):
        with self._lock:
            self.max_concurrency = max(1, int(n))
            self._ensure_workers()

    def set_bandwidth_limit(self, bps: Optional[int]):
        with self._lock:
            self.bandwidth_limit_bps = bps

    def add_job(self, job: DownloadJob):
        with self._lock:
            if job.job_id in self._jobs:
                return
            # inherit manager rate limit if job not set
            if job.bandwidth_limit_bps is None:
                job.bandwidth_limit_bps = self.bandwidth_limit_bps
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
        self._wake_workers()

    def get_jobs(self) -> List[DownloadJob]:
        with self._lock:
            return [self._jobs[j] for j in self._queue] + [j for j in self._jobs.values() if j.job_id not in self._queue and j.status in (DownloadStatus.RUNNING, DownloadStatus.PAUSED, DownloadStatus.IMPORTING)]

    def find_job(self, job_id: str) -> Optional[DownloadJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def pause_job(self, job_id: str):
        j = self.find_job(job_id)
        if j:
            j.pause()
            self.on_progress(j)

    def resume_job(self, job_id: str):
        j = self.find_job(job_id)
        if j:
            j.resume()
            with self._lock:
                if job_id not in self._queue:
                    self._queue.append(job_id)
            self._wake_workers()
            self.on_progress(j)

    def cancel_job(self, job_id: str):
        j = self.find_job(job_id)
        if j:
            j.cancel()
            self.on_progress(j)

    def stop(self):
        with self._lock:
            self._stop = True
        for t in list(self._threads):
            try:
                t.join(timeout=0.1)
            except Exception:
                pass

    def _wake_workers(self):
        # no-op placeholder to poke workers
        self._ensure_workers()

    def _ensure_workers(self):
        # spawn up to max_concurrency workers
        # remove dead threads
        alive = []
        for t in self._threads:
            if t.is_alive():
                alive.append(t)
        self._threads = alive
        while len(self._threads) < self.max_concurrency:
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._threads.append(t)

    def _pick_next_job(self) -> Optional[DownloadJob]:
        with self._lock:
            # skip paused
            for i, job_id in enumerate(list(self._queue)):
                j = self._jobs.get(job_id)
                if not j:
                    self._queue.pop(i)
                    continue
                if j.status in (DownloadStatus.QUEUED,):
                    # pop and run
                    self._queue.pop(i)
                    j.status = DownloadStatus.RUNNING
                    return j
        return None

    def _worker_loop(self):
        while True:
            if self._stop:
                return
            job = self._pick_next_job()
            if not job:
                time.sleep(0.1)
                continue
            try:
                self._run_job(job)
            except Exception as e:
                job.status = DownloadStatus.FAILED
                job.error = str(e)
                self.on_done(job)

    def _run_job(self, job: DownloadJob):
        job.start_ts = time.time()
        self.on_progress(job)
        tmp = job.dest.with_suffix(job.dest.suffix + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        resume_from = 0
        if tmp.exists():
            try:
                resume_from = tmp.stat().st_size
            except Exception:
                resume_from = 0
        headers = {}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
        with requests.get(job.url, stream=True, timeout=None, headers=headers) as r:
            if r.status_code in (206, 200):
                total = r.headers.get("Content-Length")
                # For partial content, total is remaining
                try:
                    total_val = int(total) if total else None
                except Exception:
                    total_val = None
                job.total = (resume_from + total_val) if (total_val is not None and r.status_code == 206) else (total_val or job.total)
                mode = "ab" if resume_from > 0 else "wb"
                last_tick = time.time()
                bytes_in_tick = 0
                with open(tmp, mode) as f:
                    if resume_from > 0:
                        job.downloaded = resume_from
                        self.on_progress(job)
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if job._cancel:
                            job.status = DownloadStatus.CANCELED
                            self.on_done(job)
                            return
                        # pause control
                        job._pause_ev.wait()
                        if job.status == DownloadStatus.PAUSED:
                            # yield and re-enqueue
                            with self._lock:
                                if job.job_id not in self._queue:
                                    self._queue.insert(0, job.job_id)
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        job.downloaded += len(chunk)
                        bytes_in_tick += len(chunk)
                        now = time.time()
                        if job.bandwidth_limit_bps:
                            # simple token bucket: ensure bytes_in_tick/sec <= limit
                            elapsed = now - last_tick
                            if elapsed <= 0:
                                elapsed = 1e-6
                            rate = bytes_in_tick / elapsed
                            if rate > job.bandwidth_limit_bps:
                                time.sleep((bytes_in_tick / job.bandwidth_limit_bps) - elapsed)
                        if now - job.last_update_ts > 0.25:
                            job.last_update_ts = now
                            self.on_progress(job)
                # finished download
            else:
                raise Exception(f"HTTP {r.status_code} during download")
        # rename
        tmp.rename(job.dest)
        # checksum verify if provided
        if job.sha256:
            h = hashlib.sha256()
            with open(job.dest, 'rb') as f:
                for b in iter(lambda: f.read(1024*1024), b''):
                    h.update(b)
            if h.hexdigest().lower() != job.sha256:
                job.status = DownloadStatus.FAILED
                job.error = "Checksum mismatch"
                try:
                    job.dest.unlink(missing_ok=True)
                except Exception:
                    pass
                self.on_done(job)
                return
        # import into Ollama (if provided)
        if job.ollama_import and job.suggested_name:
            try:
                job.status = DownloadStatus.IMPORTING
                self.on_progress(job)
                job.ollama_import(job.suggested_name, job.dest)
            except Exception as e:
                job.status = DownloadStatus.FAILED
                job.error = str(e)
                self.on_done(job)
                return
        job.status = DownloadStatus.DONE
        self.on_done(job)
