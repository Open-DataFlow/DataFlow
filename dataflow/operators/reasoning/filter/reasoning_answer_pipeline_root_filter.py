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
                "该算子根据样本是否具有 Ground Truth 添加答案分支标记，"
                "并尝试从答案字段中提取缺失的 Ground Truth。\n\n"
                "输入参数：\n"
                "- input_answer_key：答案字段名，默认为'output'\n"
                "- input_gt_key：Ground Truth 字段名，默认为'golden_answer'\n"
                "- output_branch_key：分支标记字段名，默认为'answer_branch'\n\n"
                "输出行为：\n"
                "- Ground Truth 存在时标记为'with_gt'\n"
                "- Ground Truth 缺失时标记为'without_gt'\n"
                "- 保留全部样本，并通过一次写入传递给下一步骤"
            )
        elif lang == "en":
            return (
                "This operator labels answer rows according to whether Ground Truth "
                "is available and attempts to extract missing Ground Truth from the "
                "answer field.\n\n"
                "Input Parameters:\n"
                "- input_answer_key: Answer column, default is 'output'\n"
                "- input_gt_key: Ground Truth column, default is 'golden_answer'\n"
                "- output_branch_key: Branch marker column, default is 'answer_branch'\n\n"
                "Output Behavior:\n"
                "- Mark rows with Ground Truth as 'with_gt'\n"
                "- Mark rows without Ground Truth as 'without_gt'\n"
                "- Preserve all rows and pass them to the next step with a single write"
            )
        else:
            return "Annotate answer rows with Ground Truth availability."

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
