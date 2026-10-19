from types import SimpleNamespace

import jinja2
import pytest

from scripts.empty_thinking_sft.prepare import fixed_template
from scripts.empty_thinking_sft.rollout import generate_rollout

TEMPLATE = r"""{%- if generation_only -%}
{{- '<think>\n\n</think>\n\n' -}}
{%- else -%}
{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}
{%- endif -%}"""


@pytest.mark.parametrize('reasoning', ['', 'A useful thought.'])
def test_only_empty_thinking_boundary_changes(reasoning: str) -> None:
    # Blank lines and marker-looking text in the actual message must survive.
    content = 'code\n\nparagraph\n<think>\n\n</think>\nunchanged'
    context = {'generation_only': False, 'message': {'role':'assistant'},
               'reasoning_content': reasoning, 'content': content}
    original = jinja2.Template(TEMPLATE).render(**context)
    actual = jinja2.Template(fixed_template(TEMPLATE)).render(**context)
    expected = original if reasoning else original.replace('<think>\n\n</think>', '<think>\n</think>', 1)
    assert actual == expected
    assert actual.endswith(content)


def test_disabled_thinking_matches_single_newline() -> None:
    assert jinja2.Template(fixed_template(TEMPLATE)).render(generation_only=True) == '<think>\n</think>\n\n'


def test_unknown_template_is_rejected() -> None:
    with pytest.raises(AssertionError, match='Unexpected assistant'):
        fixed_template('a different template')


class Buffer:
    def __init__(self) -> None:
        self.dataset = [None]*10083
        self.requested = 0

    def get_samples(self, count: int) -> list:
        self.requested = count
        return [[SimpleNamespace(metadata={'sft_tokens':[10,11,12], 'sft_mask':[0,1,1]})] for _ in range(count)]


@pytest.mark.parametrize(('step','count'), [(0,4),(2519,4),(2520,3)])
def test_final_partial_batch_and_mask(step: int, count: int) -> None:
    buffer = Buffer()
    result = generate_rollout(SimpleNamespace(rollout_batch_size=4),step,buffer)
    assert buffer.requested == len(result) == count
    assert all(group[0].tokens == [10,11,12] and group[0].loss_mask == [1,1]
               and group[0].response_length == 2 for group in result)
