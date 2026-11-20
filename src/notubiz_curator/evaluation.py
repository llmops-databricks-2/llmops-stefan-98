"""MLflow evaluation helpers for the Notubiz agent."""

import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from mlflow.genai import evaluate, scorer
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines

from notubiz_curator.agent import NotubizAgent

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "databricks:/course_ops.gateway.chat")
MAX_WORDS = 350

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response uses a polite and professional tone throughout.",
        "The response is helpful and concise without being condescending.",
        "The response avoids dismissive, rude, or unnecessarily defensive language.",
    ],
    model=JUDGE_MODEL,
)

scope_guideline = Guidelines(
    name="stays_in_notubiz_scope",
    guidelines=[
        "The response stays within questions about Notubiz and public Wetterskip Fryslân documents.",
        "For unrelated questions, the response politely explains the knowledge boundary and does not guess.",
        "The response does not present general knowledge or invented facts as Notubiz document evidence.",
    ],
    model=JUDGE_MODEL,
)

notubiz_answer_judge = make_judge(
    name="notubiz_answer_quality",
    instructions=(
        "Evaluate {{ outputs }} for the user request in {{ inputs }} against the reviewed expectations. "
        "The expectations contain reference_answer, expected_behavior, and forbidden_behavior in {{ expectations }}. "
        "Pass only when the response fulfills the expected behavior, does not exhibit forbidden behavior, "
        "and is appropriately grounded in the evidence or honestly explains a knowledge boundary. "
        "For document-specific claims, require actual source URLs or equivalent retrieved evidence when available. "
        "Do not require unsupported exact content when the supplied expectation calls for clarification or refusal. "
        "Return pass when the answer is appropriate, fail when it is materially wrong or violates the task, and "
        "insufficient_evidence only when the provided expectations or retrieved context cannot support a fair verdict."
    ),
    feedback_value_type=Literal["pass", "fail", "insufficient_evidence"],
    model=JUDGE_MODEL,
)


def load_eval_data(eval_inputs_path: str | Path) -> list[dict[str, Any]]:
    """Load reviewed JSONL evaluation cases."""
    cases: list[dict[str, Any]] = []
    with open(eval_inputs_path, encoding="utf-8") as eval_file:
        for line_number, line in enumerate(eval_file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case, dict) or not isinstance(case.get("inputs"), dict):
                raise ValueError(f"Evaluation case on line {line_number} must contain an inputs object")
            if not isinstance(case["inputs"].get("question"), str):
                raise ValueError(f"Evaluation case on line {line_number} must contain an input question")
            if not isinstance(case.get("expectations"), dict):
                raise ValueError(f"Evaluation case on line {line_number} must contain an expectations object")
            cases.append(cast(dict[str, Any], case))
    if not cases:
        raise ValueError("Evaluation dataset must contain at least one reviewed case")
    return cases


@scorer
def concise_nonempty_answer(outputs: object) -> bool:
    """Check that the final answer is non-empty and no longer than 350 words."""
    text = _output_text(outputs)
    word_count = len(text.split())
    return 0 < word_count <= MAX_WORDS


def evaluate_agent(agent: NotubizAgent, eval_inputs_path: str | Path) -> EvaluationResult:
    """Evaluate the initialized agent on reviewed inputs."""
    eval_data = load_eval_data(eval_inputs_path)

    def predict_fn(question: str, history: list[dict[str, Any]] | None = None) -> str:
        return agent.generate_message(question, history=history)  # type: ignore[arg-type]

    return evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            concise_nonempty_answer,
            polite_tone_guideline,
            scope_guideline,
            notubiz_answer_judge,
        ],
    )


def _output_text(outputs: object) -> str:
    """Normalize common MLflow/OpenAI output shapes to plain text."""
    if isinstance(outputs, str):
        return outputs.strip()
    if isinstance(outputs, list):
        return " ".join(_output_text(item) for item in cast(list[object], outputs)).strip()
    if isinstance(outputs, dict):
        output_dict = cast(dict[str, object], outputs)
        if "text" in output_dict:
            return _output_text(output_dict["text"])
        if "content" in output_dict:
            return _output_text(output_dict["content"])
    return str(outputs).strip()
