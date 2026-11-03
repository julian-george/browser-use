"""Beaker integration: the browser-use Agent on BU Bench V2.1, scored by the benchmark's findings judge.

Mirrors the benchmark's run_eval.run_task (browser-use executor) on this fork's code. See README.md.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import bu_bench
from beaker import (
	Case,
	CaseResult,
	CaseScore,
	Check,
	Integration,
	JsonValue,
	RepositoryRunSetupResult,
	RetryableCaseError,
	RolloutRuntime,
	RunSetup,
	SetupRuntime,
	inference_target,
	repository,
	scoring_inference_target,
)
from pydantic import BaseModel, ConfigDict, Field
from traced_chat import TracedChatOpenAI

AGENT_MODEL = 'gpt-6-luna'
JUDGE_MODEL = 'gpt-6-luna'  # Must match the launch scorer_model (openai:gpt-6-luna).
DEFAULT_TASK_TIMEOUT_S = 1800
MAX_STEPS = 100
# Hosted judge calls go straight to Beaker's gateway, which 413s large bodies (upstream allows 8 MB).
# Screenshots get what the text evidence leaves of this; it also keeps the case result under 8 MiB.
JUDGE_REQUEST_LIMIT_BYTES = 4 * 1024 * 1024
JUDGE_PROMPT_HEADROOM_BYTES = 256 * 1024
SHORT_CHROMIUM_TMPDIR = '/tmp/bu-chromium'

EDITABLE_PATHS = (
	'browser_use/agent/system_prompts/system_prompt.md',
	'browser_use/agent/prompts.py',
	'browser_use/agent/message_manager',
	'browser_use/dom/serializer',
	'browser_use/tools',
	'browser_use/agent/service.py',
)


class Row(BaseModel):
	model_config = ConfigDict(extra='forbid')

	id: str = Field(min_length=1)
	input: TaskRef
	expected: RubricRef
	metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TaskRef(BaseModel):
	model_config = ConfigDict(extra='forbid')

	task_id: str
	revision: str | None
	task_sha: str


class RubricRef(BaseModel):
	model_config = ConfigDict(extra='forbid')

	rubric_sha: str
	weights_sha: str


Row.model_rebuild()


class Setup(RunSetup[Row]):
	row_model = Row
	_tasks_by_id: dict[str, dict]

	@asynccontextmanager
	async def prepare_run(self, *, runtime: SetupRuntime) -> AsyncIterator[dict[str, RepositoryRunSetupResult]]:
		# Decrypt once per attempt; rows load concurrently.
		self._tasks_by_id = await asyncio.to_thread(bu_bench.load_tasks_by_id)
		yield {'repository': RepositoryRunSetupResult()}

	async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
		task = self._tasks_by_id[row.input.task_id]
		identity = {
			'revision': (task.get('revision'), row.input.revision),
			'task_sha': (task['task_sha'], row.input.task_sha),
			'rubric_sha': (task['rubric_sha'], row.expected.rubric_sha),
			'weights_sha': (task['weights_sha'], row.expected.weights_sha),
		}
		mismatched = [name for name, (pinned, from_row) in identity.items() if pinned != from_row]
		# The shas are identity stamps (not all re-derivable); file integrity comes from the pinned .enc sha256.
		if mismatched:
			raise ValueError(f'{row.id}: pinned benchmark task does not match the dataset row ({mismatched})')
		case_input: dict[str, JsonValue] = {
			'task_id': task['task_id'],
			'instruction': task['confirmed_task'],
			'task_timeout_s': int(runtime.config.get('task_timeout_s', DEFAULT_TASK_TIMEOUT_S)),
		}
		if task.get('website'):
			case_input['website'] = task['website']
		# The rubric only reaches the scorer; the candidate process never sees it.
		judge_task_fields = ('task_id', 'confirmed_task', 'rubric', 'weights', 'canary', 'website')
		yield Case(
			id=row.id,
			input=case_input,
			expected={field: task.get(field) for field in judge_task_fields},
			metadata=row.metadata,
		)


def _agent_llm(runtime: RolloutRuntime) -> TracedChatOpenAI:
	if runtime.model:
		target = inference_target(runtime)
		model, base_url, api_key = target.model, target.base_url, target.api_key
	else:
		model, base_url, api_key = AGENT_MODEL, None, os.getenv('OPENAI_API_KEY')
	# Upstream's create_agent_model settings; reasoning_models makes the fork send reasoning_effort.
	return TracedChatOpenAI(
		model=model,
		base_url=base_url,
		api_key=api_key,
		reasoning_effort='xhigh',
		reasoning_models=[model],
		max_completion_tokens=32768,
		temperature=None,
		frequency_penalty=None,
		timeout=300,
		max_retries=2,
	)


def _encode_screenshots(paths: list[str], evidence_notes: list[str]) -> list[str]:
	images = []
	for index, path in enumerate(paths, 1):
		try:
			images.append(base64.b64encode(Path(path).read_bytes()).decode())
		except OSError:
			evidence_notes.append(f'Capture {index} could not be read from the saved screenshots.')
	return images


def _collect_output_files(agent, evidence_notes: list[str]) -> str:
	contents = []
	for name in agent.file_system.list_files():
		try:
			contents.append(f'--- {name} ---\n{agent.file_system.get_file(name).read()}')
		except Exception:
			evidence_notes.append(f'Managed output file {name} could not be read by the collector.')
	return '\n\n'.join(contents)


def _record_step_actions(trace, agent) -> None:
	step = agent.history.history[-1] if agent.history.history else None
	if step is None or step.model_output is None:
		return
	for index, (action, result) in enumerate(zip(step.model_output.action, step.result)):
		arguments = action.model_dump(exclude_none=True, mode='json')
		name = next(iter(arguments), 'action')
		with trace.tool_call(
			name, arguments=arguments[name] if name in arguments else arguments, call_id=f'{agent.state.n_steps}-{index}'
		) as call:
			call.output({'extracted_content': result.extracted_content, 'error': result.error, 'is_done': result.is_done})


async def _chromium_stderr(browser) -> str:
	"""Relaunch Chromium with the session's flags to surface why it exits; browser-use discards its stderr."""
	executable = browser.browser_profile.executable_path
	if executable is None:
		return 'no explicit executable to probe'
	process = await asyncio.create_subprocess_exec(
		str(executable),
		*browser.browser_profile.get_args(),
		'--remote-debugging-port=0',
		'about:blank',
		stdout=asyncio.subprocess.DEVNULL,
		stderr=asyncio.subprocess.PIPE,
	)
	try:
		_, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
	except TimeoutError:
		process.kill()
		return 'probe stayed up for 15s'
	lines = [line for line in stderr.decode(errors='replace').splitlines() if line.strip() and 'dbus' not in line]
	return f'exit {process.returncode}, uid {os.geteuid()}: ' + ' / '.join(lines[-4:])[-600:]


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
	from browser_use import Agent, Browser

	assert isinstance(case_input, dict)
	llm = _agent_llm(runtime)
	profile_dir = tempfile.mkdtemp(prefix='bu-bench-profile-')
	if sys.platform == 'linux':
		# Beaker's deep candidate TMPDIR overflows Chromium's SingletonSocket path limit (~108 chars).
		os.makedirs(SHORT_CHROMIUM_TMPDIR, exist_ok=True)
		os.environ['TMPDIR'] = SHORT_CHROMIUM_TMPDIR
	browser = Browser(
		headless=True,
		executable_path=shutil.which('chromium'),  # Hosted image's apt Chromium; None locally uses browser-use's default.
		user_data_dir=profile_dir,
		chromium_sandbox=sys.platform != 'linux',  # The hosted Linux sandbox cannot host Chromium's own sandbox.
	)
	execution_error = None
	try:
		try:
			await browser.start()
		except Exception as exc:
			# Probe output first: Beaker truncates long case errors.
			raise RetryableCaseError(f'Chromium did not start: {await _chromium_stderr(browser)} ({exc})') from exc
		agent = Agent(task=case_input['instruction'], llm=llm, page_extraction_llm=llm, browser=browser, use_judge=False)

		async def on_step_end(agent_after_step) -> None:
			_record_step_actions(runtime.trace, agent_after_step)

		with runtime.trace.stage('browser_use.agent.run', inputs={'task_id': case_input['task_id']}):
			try:
				history = await asyncio.wait_for(
					agent.run(max_steps=MAX_STEPS, on_step_end=on_step_end), timeout=case_input['task_timeout_s']
				)
			except TimeoutError:
				history = agent.history  # Upstream judges partial work after a timeout.
				execution_error = f'Task timed out after {case_input["task_timeout_s"]}s'
			except Exception as exc:
				history = agent.history
				execution_error = f'{type(exc).__name__}: {exc}'

		if not any(step.model_output is not None for step in history.history):
			# Browser launch, provider or CDP failure before the agent took a step: not an agent score.
			errors = [error for error in history.errors() if error]
			raise RetryableCaseError(f'Agent never produced a step: {execution_error or errors[-1:] or "no error recorded"}')

		evidence_notes: list[str] = []
		agent_steps = history.agent_steps()
		output_files_text = _collect_output_files(agent, evidence_notes)
		raw_screenshots = _encode_screenshots([path for path in history.screenshot_paths() if path is not None], evidence_notes)
	finally:
		await browser.stop()
		shutil.rmtree(profile_dir, ignore_errors=True)

	output = {
		'final_result': history.final_result() or 'Agent did not return a result',
		'is_done': history.is_done(),
		'steps': history.number_of_steps(),
		'duration_s': history.total_duration_seconds(),
		'agent_cost': history.usage.total_cost if history.usage else 0,
		'execution_error': execution_error,
	}
	text_bytes = len(json.dumps([output, agent_steps, output_files_text, evidence_notes]))
	screenshot_budget = max(500_000, JUDGE_REQUEST_LIMIT_BYTES - JUDGE_PROMPT_HEADROOM_BYTES - text_bytes)
	screenshots, screenshot_metadata = bu_bench.screenshot_evidence().prepare_screenshots(raw_screenshots, 50, screenshot_budget)
	evidence_notes.extend(screenshot_metadata['notes'])
	return CaseResult(
		output=output,
		output_kind='record',
		context={
			'agent_steps': agent_steps,
			'screenshots_b64': screenshots,
			'output_files_text': output_files_text,
			'evidence_notes': evidence_notes,
		},
	)


def _judge_llm():
	from browser_use.llm import ChatOpenAI

	target = scoring_inference_target()
	if target is None:
		model, base_url, api_key = JUDGE_MODEL, None, os.getenv('OPENAI_API_KEY')
	else:
		model, base_url, api_key = target.model, target.base_url, target.api_key
	# evaluation.create_judge's settings, plus reasoning_models so the fork sends reasoning_effort.
	return ChatOpenAI(
		model=model,
		base_url=base_url,
		api_key=api_key,
		reasoning_effort='xhigh',
		reasoning_models=[model],
		max_completion_tokens=32768,
		temperature=None,
		frequency_penalty=None,
		timeout=300,
		max_retries=2,
	)


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
	del case_files_dir
	if not result.context:
		raise RuntimeError('Judge evidence (CaseResult.context) was dropped; refusing to score without it')
	task = dict(case.expected)  # type: ignore[arg-type]
	output = result.output
	assert isinstance(output, dict)
	trace = {'final_result': output['final_result'], **result.context}
	judged = await bu_bench.evaluation().judge_trace(task, trace, _judge_llm())

	findings = {finding['item']: finding for finding in judged['judgement']['findings']}
	checks = [
		Check(
			name=item,
			verdict='pass' if findings[item]['status'] == 'met' else 'fail',
			message=f'{findings[item]["status"]} ({findings[item]["not_assessable_reason"] or "-"}): {findings[item]["evidence"]}',
			group=f'weight {weight}',
		)
		for item, weight in task['weights'].items()
	]
	if judged['rh_zeroed']:
		reason = 'canary leaked' if judged['canary_leak'] else judged['judgement']['flag_notes']
		checks.append(Check(name='reward_hacking', verdict='fail', message=f'Score zeroed: {reason}'))
	if judged['judgement']['infra_error']:
		checks.append(
			Check(name='infra_error', verdict='fail', informational=True, message=judged['judgement']['flag_notes'] or '')
		)
	return CaseScore(
		objective=judged['score'],
		field_scores={'score': judged['score'], 'raw_score': judged['raw_score']},
		checks=tuple(checks),
	)


integration = Integration(
	targets=repository(EDITABLE_PATHS),
	run_setup=Setup,
	run_case=run_case,
	score_case=score_case,
)
