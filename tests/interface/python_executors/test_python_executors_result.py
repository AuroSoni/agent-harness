"""Red-suite specs for ``ExecutorResult`` (was ``CodeOutput``).

Covers python-executors.md:
  - §2.3 ``ExecutorResult`` dataclass: positional ``output``/``logs``/
    ``is_final_answer`` + NEW ``truncated: bool = False`` and
    ``error: InterpreterError | None = None``.
  - §6 migration: ``CodeOutput`` alias is DELETED (G0); new fields default so
    construction is unchanged.
  - §3.2: ``result.truncated`` flag lets the tool skip re-truncation (F6).

Imports target ``agent_base.python_executors.local_python_executor`` (the doc's
stated source for the executor value types). The implementation does not exist
yet.
"""

import dataclasses

import agent_base.python_executors.local_python_executor as lpe
from agent_base.python_executors.base import InterpreterError
from agent_base.python_executors.local_python_executor import ExecutorResult


def test_executor_result_is_a_dataclass():
    assert dataclasses.is_dataclass(ExecutorResult)


def test_executor_result_field_names_and_order():
    names = [f.name for f in dataclasses.fields(ExecutorResult)]
    assert names == ["output", "logs", "is_final_answer", "truncated", "error"]


def test_executor_result_minimal_construction_defaults():
    result = ExecutorResult(output=42, logs="hello", is_final_answer=False)
    assert result.output == 42
    assert result.logs == "hello"
    assert result.is_final_answer is False
    assert result.truncated is False
    assert result.error is None


def test_executor_result_truncated_flag_settable():
    result = ExecutorResult(output=None, logs="x" * 10, is_final_answer=False, truncated=True)
    assert result.truncated is True


def test_executor_result_carries_structured_error():
    err = InterpreterError("boom")
    result = ExecutorResult(output=None, logs="", is_final_answer=False, error=err)
    assert result.error is err
    assert isinstance(result.error, InterpreterError)


def test_executor_result_final_answer_value():
    result = ExecutorResult(output="answer", logs="", is_final_answer=True)
    assert result.is_final_answer is True
    assert result.output == "answer"


def test_code_output_alias_is_deleted():
    # §6 (G0): CodeOutput = ExecutorResult alias removed.
    assert not hasattr(lpe, "CodeOutput")
