import importlib
import pandas as pd
import pytest

from dataflow.operators.reasoning import ReasoningAnswerPipelineRootFilter
from dataflow.utils.storage import DataFlowStorage


class SpyStorage(DataFlowStorage):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self.result = None
        self.write_count = 0

    def get_keys_from_dataframe(self) -> list[str]:
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        if output_type == "dataframe":
            return self.dataframe.copy()
        if output_type == "dict":
            return self.dataframe.to_dict("records")
        raise ValueError(f"Unsupported output type: {output_type}")

    def write(self, data):
        self.write_count += 1
        self.result = data.copy()
        return "memory://reasoning-root-filter"


@pytest.mark.cpu
def test_root_filter_writes_once_and_preserves_both_groups():
    storage = SpyStorage(
        pd.DataFrame(
            [
                {"instruction": "q1", "golden_answer": "1"},
                {"instruction": "q2", "golden_answer": None},
            ]
        )
    )

    operator = ReasoningAnswerPipelineRootFilter()
    result_keys = operator.run(
        storage=storage,
        input_answer_key="missing_answer",
        input_gt_key="golden_answer",
        output_branch_key="answer_branch",
    )

    assert storage.write_count == 1
    assert len(storage.result) == 2
    assert storage.result["answer_branch"].tolist() == [
        "with_gt",
        "without_gt",
    ]
    assert result_keys == ["golden_answer", "answer_branch"]


@pytest.mark.cpu
def test_root_filter_creates_missing_gt_column_and_still_writes():
    storage = SpyStorage(
        pd.DataFrame(
            [
                {"instruction": "q1"},
                {"instruction": "q2"},
            ]
        )
    )

    operator = ReasoningAnswerPipelineRootFilter()
    operator.run(
        storage=storage,
        input_answer_key="missing_answer",
        input_gt_key="golden_answer",
        output_branch_key="answer_branch",
    )

    assert storage.write_count == 1
    assert "golden_answer" in storage.result.columns
    assert storage.result["golden_answer"].isna().all()
    assert (storage.result["answer_branch"] == "without_gt").all()


@pytest.mark.cpu
def test_root_filter_marks_all_existing_gt_rows():
    storage = SpyStorage(
        pd.DataFrame(
            [
                {"golden_answer": "1"},
                {"golden_answer": "2"},
            ]
        )
    )

    operator = ReasoningAnswerPipelineRootFilter()
    operator.run(
        storage=storage,
        input_answer_key="missing_answer",
        input_gt_key="golden_answer",
        output_branch_key="answer_branch",
    )

    assert storage.write_count == 1
    assert len(storage.result) == 2
    assert (storage.result["answer_branch"] == "with_gt").all()


@pytest.mark.cpu
def test_root_filter_extracts_missing_gt_from_answer(monkeypatch):
    def fake_extract_answer(self, answer, *_args, **_kwargs):
        return "42" if answer else None

    root_filter_module = importlib.import_module(
        "dataflow.operators.reasoning.filter.reasoning_answer_pipeline_root_filter"
    )
    def fake_extract_answer(self, answer, *_args, **_kwargs):
        return "42" if answer else None

    monkeypatch.setattr(
        root_filter_module.AnswerExtractor, "extract_answer", fake_extract_answer,
    )

    storage = SpyStorage(
        pd.DataFrame(
            [
                {
                    "output": "The final answer is 42.",
                    "golden_answer": None,
                }
            ]
        )
    )

    operator = ReasoningAnswerPipelineRootFilter()
    operator.run(storage=storage)

    assert storage.write_count == 1
    assert storage.result.loc[0, "golden_answer"] == "42"
    assert storage.result.loc[0, "answer_branch"] == "with_gt"
