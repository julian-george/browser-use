from __future__ import annotations

from typing import Any

from beaker.tracing import current_trace

from browser_use.llm import ChatOpenAI
from browser_use.llm.messages import BaseMessage


class TracedChatOpenAI(ChatOpenAI):
	"""ChatOpenAI that records each call in the active Beaker capture.

	No built-in Beaker integration covers browser-use's own clients; a subclass keeps the type the Agent expects.
	"""

	async def ainvoke(self, messages: list[BaseMessage], output_format: Any = None, **kwargs: Any) -> Any:
		with current_trace().model_call(
			operation='chat',
			provider='openai',
			model=self.model,
			input_messages=[message.model_dump(mode='json') for message in messages],
		) as call:
			response = await super().ainvoke(messages, output_format, **kwargs)
			completion = response.completion
			call.output(completion.model_dump(mode='json') if hasattr(completion, 'model_dump') else completion)
			if response.usage is not None:
				call.usage(
					input_tokens=response.usage.prompt_tokens,
					output_tokens=response.usage.completion_tokens,
					cached_input_tokens=response.usage.prompt_cached_tokens,
				)
			return response
