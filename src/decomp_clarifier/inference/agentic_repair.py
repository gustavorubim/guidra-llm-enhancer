from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from decomp_clarifier.dataset.prompt_formatter import format_prompt
from decomp_clarifier.inference.checkpoint_predictor import CheckpointPredictor
from decomp_clarifier.inference.formatter import normalize_output_with_status
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord
from decomp_clarifier.training.grpo.verifier import VerificationResult, verify_output

PromptFormatter = Callable[[FunctionDatasetSample], str]

_FULL_NO_THINKING_LINE = "Do not include markdown fences, commentary, XML tags, or <think> blocks."
_COMPACT_NO_THINKING_LINE = "Do not include markdown, commentary, XML tags, or <think> blocks."
_THINKING_LINE = (
    "You may use the model thinking channel, but the final answer after </think> "
    "must be exactly one JSON object. Do not include markdown fences, commentary, "
    "or XML tags in the final answer."
)


@dataclass(frozen=True)
class AgenticAttempt:
    attempt_index: int
    raw_text: str
    json_valid: bool
    field_complete: bool
    compile_success: bool
    behavior_success: bool
    feedback: list[str]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgenticPrediction:
    record: PredictionRecord
    attempts: list[AgenticAttempt]


def agentic_prompt(prompt: str, *, enable_thinking: bool) -> str:
    if not enable_thinking:
        return prompt
    return prompt.replace(_FULL_NO_THINKING_LINE, _THINKING_LINE).replace(
        _COMPACT_NO_THINKING_LINE,
        _THINKING_LINE,
    )


def _truncate(text: str, max_chars: int = 2400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def build_repair_prompt(
    *,
    original_prompt: str,
    previous_answer: str,
    feedback: list[str],
    attempt_index: int,
) -> str:
    feedback_lines = "\n".join(f"- {item}" for item in feedback)
    return "\n".join(
        [
            original_prompt,
            "",
            "Tool validation failed for your previous answer.",
            f"Repair attempt: {attempt_index}",
            "",
            "Tool feedback:",
            feedback_lines,
            "",
            "Previous answer:",
            "<answer>",
            _truncate(previous_answer),
            "</answer>",
            "",
            "Return a corrected final answer. The final answer must be exactly one JSON object "
            "with keys summary, confidence, renamings, cleaned_c.",
            "JSON:",
        ]
    )


def _invalid_verification() -> VerificationResult:
    return VerificationResult(
        json_valid=False,
        field_complete=False,
        compile_success=False,
        behavior_success=False,
        readability_score=0.0,
        naming_score=0.0,
        placeholder_ratio=0.0,
    )


def validate_agentic_answer(
    sample: FunctionDatasetSample,
    raw_text: str,
    *,
    strip_thinking: bool,
) -> tuple[ClarifiedFunctionOutput, bool, VerificationResult, list[str]]:
    output, json_valid = normalize_output_with_status(raw_text, strip_thinking=strip_thinking)
    if not json_valid:
        return (
            output,
            False,
            _invalid_verification(),
            [
                "The answer was not strict JSON after removing any thinking prefix. "
                "Return exactly one parseable JSON object and no extra final text."
            ],
        )
    verification = verify_output(sample, output, json_valid=True)
    feedback: list[str] = []
    if not verification.field_complete:
        feedback.append("The JSON object is missing non-empty required fields.")
    if not verification.compile_success:
        feedback.append(
            "The cleaned_c field failed the compile check. Preserve the original function "
            "signature and return valid C."
        )
    if not verification.behavior_success:
        feedback.append(
            "The cleaned_c field failed the behavior/semantic check. Preserve the original "
            "function semantics while clarifying names and structure."
        )
    if verification.placeholder_ratio > 0.15:
        feedback.append(
            "The cleaned_c field still contains too many decompiler placeholders such as "
            "param_*, local_*, iVar*, or uVar*."
        )
    return output, True, verification, feedback


class AgenticRepairPredictor:
    def __init__(
        self,
        predictor: CheckpointPredictor,
        *,
        prompt_formatter: PromptFormatter = format_prompt,
        max_repair_attempts: int = 2,
    ) -> None:
        self.predictor = predictor
        self.prompt_formatter = prompt_formatter
        self.max_repair_attempts = max_repair_attempts

    def predict(
        self,
        sample: FunctionDatasetSample,
        *,
        system: str,
        max_new_tokens: int,
        temperature: float,
    ) -> AgenticPrediction:
        original_prompt = agentic_prompt(
            self.prompt_formatter(sample),
            enable_thinking=self.predictor.enable_thinking,
        )
        prompt = original_prompt
        attempts: list[AgenticAttempt] = []
        output: ClarifiedFunctionOutput | None = None
        json_valid = False
        for attempt_index in range(self.max_repair_attempts + 1):
            raw_text = self.predictor.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            output, json_valid, verification, feedback = validate_agentic_answer(
                sample,
                raw_text,
                strip_thinking=self.predictor.enable_thinking,
            )
            attempts.append(
                AgenticAttempt(
                    attempt_index=attempt_index,
                    raw_text=raw_text,
                    json_valid=json_valid,
                    field_complete=verification.field_complete,
                    compile_success=verification.compile_success,
                    behavior_success=verification.behavior_success,
                    feedback=feedback,
                )
            )
            if not feedback:
                break
            prompt = build_repair_prompt(
                original_prompt=original_prompt,
                previous_answer=raw_text,
                feedback=feedback,
                attempt_index=attempt_index + 1,
            )
        if output is None:
            raise RuntimeError("agentic predictor did not produce an attempt")
        return AgenticPrediction(
            record=PredictionRecord(
                sample_id=sample.sample_id,
                system=system,
                output=output,
                raw_text=attempts[-1].raw_text,
                json_valid=json_valid,
            ),
            attempts=attempts,
        )
