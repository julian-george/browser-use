"""Build ids-only BU Bench V2.1 rows from splits.json in a temp dir, then smoke or upload them.

Usage: uv run python .beaker/build_dataset.py {full|quickstart} {smoke|upload}
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bu_bench

AGENT_KEY = 'browser-use-bu-bench-v2-1'
DATASET_NAMES = {'full': 'bu-bench-v2-1', 'quickstart': 'bu-bench-v2-1-quickstart'}


def build_rows(task_ids: list[str], tasks_by_id: dict[str, dict]) -> list[dict]:
	return [
		{
			'id': task_id,
			'input': {
				'task_id': task_id,
				'revision': tasks_by_id[task_id].get('revision'),
				'task_sha': tasks_by_id[task_id]['task_sha'],
			},
			'expected': {
				'rubric_sha': tasks_by_id[task_id]['rubric_sha'],
				'weights_sha': tasks_by_id[task_id]['weights_sha'],
			},
		}
		for task_id in task_ids
	]


def main(size: str, action: str) -> None:
	splits = json.loads((Path(__file__).parent / 'splits.json').read_text())
	split_ids = splits['quickstart'] if size == 'quickstart' else splits
	tasks_by_id = bu_bench.load_tasks_by_id()
	rows = {name: build_rows(split_ids[name], tasks_by_id) for name in ('train', 'test')}
	with tempfile.TemporaryDirectory(prefix='beaker-dataset-') as temp_dir:
		for name, split_rows in rows.items():
			(Path(temp_dir) / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in split_rows))
		if action == 'smoke':
			command = ['beaker', 'run', 'smoke', '--strict', '--config', json.dumps({'local_dataset_path': temp_dir})]
		else:
			command = [
				'beaker', 'dataset', 'upload', temp_dir,
				'--name', DATASET_NAMES[size],
				'--total-count', str(sum(len(split_rows) for split_rows in rows.values())),
				'--split', f'train={len(rows["train"])}',
				'--split', f'test={len(rows["test"])}',
				'--agent', AGENT_KEY,
			]  # fmt: skip
		subprocess.run(command, check=True)


if __name__ == '__main__':
	main(*sys.argv[1:3])
