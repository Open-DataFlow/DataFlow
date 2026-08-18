from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage

import pandas as pd

@OPERATOR_REGISTRY.register()
class ReasoningAnswerPipelineRootFilter(OperatorABC):
    def __init__(self):

        self.logger = get_logger()
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "根据样本是否具有 Ground Truth 对答案数据进行路由标记。"
                "算子会尝试从答案字段补充缺失的 Ground Truth，并在输出字段中写入 "
                "'with_gt' 或 'without_gt'。所有记录通过一次写入传递给下一步骤。"
            )
        if lang == "en":
            return (
                "Annotate answer rows according to whether Ground Truth exists. "
                "Missing Ground Truth may be extracted from the answer column. "
                "The output branch column contains 'with_gt' or 'without_gt', "
                "and all rows are persisted with a single storage write."
            )
        return "Annotate answer rows with Ground Truth routing information."

    def run(
        self,
        storage: DataFlowStorage,
        input_answer_key: str = "output",
        input_gt_key: str = "golden_answer",
        output_branch_key: str = "answer_branch",
    ) -> list[str]:
        """Annotate rows according to whether a ground-truth answer exists.

        The current DataFlow storage contract is linear: one operator step writes
        one downstream dataset. Therefore this operator records the routing result
        in ``output_branch_key`` and performs exactly one storage write.
        """
        dataframe = storage.read("dataframe").copy()

        if not input_gt_key:
            self.logger.warning(
                "No valid gt key provided; falling back to 'golden_answer'."
            )
            input_gt_key = "golden_answer"

        if input_gt_key not in dataframe.columns:
            self.logger.warning(
                "Ground-truth column '%s' is missing; creating an empty column.",
                input_gt_key,
            )
            dataframe[input_gt_key] = None

        if input_answer_key in dataframe.columns:
            unit_text_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_text_manager)
            answer_extractor = AnswerExtractor(string_cleaner)

            def resolve_ground_truth(row):
                ground_truth = row[input_gt_key]
                if pd.notna(ground_truth) and ground_truth != "":
                    return ground_truth

                answer = row[input_answer_key]
                if pd.isna(answer) or answer == "":
                    return None

                try:
                    return answer_extractor.extract_answer(answer, None, True)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to extract ground truth from '%s': %s",
                        input_answer_key,
                        exc,
                    )
                    return None

            dataframe[input_gt_key] = dataframe.apply(
                resolve_ground_truth,
                axis=1,
            )
        else:
            self.logger.warning(
                "Answer column '%s' is missing; existing ground-truth values "
                "will be used without answer extraction.",
                input_answer_key,
            )

        has_ground_truth = (
            dataframe[input_gt_key].notna()
            & dataframe[input_gt_key].ne("")
        )

        dataframe[output_branch_key] = "without_gt"
        dataframe.loc[has_ground_truth, output_branch_key] = "with_gt"

        output_file = storage.write(dataframe)

        with_gt_count = int(has_ground_truth.sum())
        without_gt_count = len(dataframe) - with_gt_count
        self.logger.info(
            "Saved %d rows to %s: with_gt=%d, without_gt=%d",
            len(dataframe),
            output_file,
            with_gt_count,
            without_gt_count,
        )

        return [input_gt_key, output_branch_key]