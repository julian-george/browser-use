"""Fetch BU Bench V2.1's pinned dataset and judge at runtime; nothing from it is committed.

The benchmark repo is unlicensed and asks that decrypted tasks stay private, so this
module downloads the pinned files into a cache, verifies their sha256, and imports them.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import httpx

BENCHMARK_COMMIT = 'af6c7f7f6772b6985b7644f660cac87fd4b03583'  # release v2.1.1
PINNED_FILE_SHA256 = {
	'BU_Bench_V2.enc': '0c014b056192e261b06f04ed6da6dcf37f2135150cf58c87ea839bd4e150e5c3',
	'evaluation.py': '70c916b571cae8f205f81e6592c6fb54d3f10222f000e2e8d66780b62bdcac09',
	'findings_judge.py': '97f659202fa15bc6702ef54132ee46e95abce0d4d854fa0cef68186151cdded6',
	'judge.py': '012d20a030383ce668f47a4c85bd428d933413471d18188fc5b599fe69e70333',
	'screenshot_evidence.py': '277907eee354c6d5635444820d8c62f9aa16fa6ed7a25e8dba464a7e41c2f0c0',
}
CACHE_DIR = Path(tempfile.gettempdir()) / 'bu-bench' / BENCHMARK_COMMIT


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_pinned_files() -> None:
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	for name, expected_sha in PINNED_FILE_SHA256.items():
		path = CACHE_DIR / name
		if path.exists() and _sha256(path) == expected_sha:
			continue
		url = f'https://raw.githubusercontent.com/browser-use/benchmark/{BENCHMARK_COMMIT}/{name}'
		response = httpx.get(url, timeout=60, follow_redirects=True)
		response.raise_for_status()
		if hashlib.sha256(response.content).hexdigest() != expected_sha:
			raise RuntimeError(f'{name} from {url} does not match its pinned sha256')
		# Unique staging file + atomic replace: concurrent fetchers may race on the same file.
		descriptor, staging = tempfile.mkstemp(dir=CACHE_DIR, prefix=f'{name}.', suffix='.part')
		with os.fdopen(descriptor, 'wb') as staging_file:
			staging_file.write(response.content)
		os.replace(staging, path)


def _import_pinned(module_name: str) -> ModuleType:
	_ensure_pinned_files()
	if str(CACHE_DIR) not in sys.path:
		sys.path.insert(0, str(CACHE_DIR))
	module = importlib.import_module(module_name)
	# The benchmark uses generic top-level names; refuse a same-named module from elsewhere.
	if Path(module.__file__ or '').parent != CACHE_DIR:
		raise RuntimeError(f'Imported {module_name} from {module.__file__}, not the pinned benchmark')
	return module


def evaluation() -> ModuleType:
	return _import_pinned('evaluation')


def screenshot_evidence() -> ModuleType:
	return _import_pinned('screenshot_evidence')


def load_tasks_by_id() -> dict[str, dict]:
	"""Decrypt V2 in memory with the benchmark's own loader, keyed by task id."""
	return {task['task_id']: task for task in evaluation().load_tasks('BU_Bench_V2')}
