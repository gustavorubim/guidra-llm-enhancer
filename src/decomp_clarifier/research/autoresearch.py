from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from decomp_clarifier.inference.formatter import normalize_output_with_status
from decomp_clarifier.paths import ProjectPaths

DEFAULT_PROFILE = "grpo_qwen35_2b"
STOP_SECTION = "## STOP"
EXPERIMENT_SURFACE = (
    "configs/training/grpo_qwen35_2b.yaml",
    "src/decomp_clarifier/training/grpo/rewards.py",
    "src/decomp_clarifier/training/grpo/train.py",
    "src/decomp_clarifier/training/grpo/data.py",
    "src/decomp_clarifier/dataset/prompt_formatter.py",
    "src/decomp_clarifier/dataset/packers.py",
)
EVAL_SURFACE = ("src/decomp_clarifier/evaluation",)
CONSISTENCY_SURFACE = (*EXPERIMENT_SURFACE, *EVAL_SURFACE)
SAFE_DIRTY_PREFIXES = (
    "src/decomp_clarifier/research/",
    "tests/unit/test_autoresearch.py",
    "research/baseline.json",
    "research/experiment_log.jsonl",
    "artifacts/logs/",
)

_SCORE_KEYS = (
    "json_valid_rate",
    "field_complete_rate",
    "readability_score",
    "compile_success_rate",
    "behavior_success_rate",
    "naming_score",
)
_BASELINE_KEYS = (
    "score",
    *_SCORE_KEYS,
    "iteration",
    "train_run_id",
    "eval_run_id",
    "timestamp",
    "config_snapshot",
)

_PROGRAM_STATUS_RE = re.compile(r"^Status:\s*(?P<status>\w+)\s*$", re.MULTILINE)
_PROGRAM_REASON_RE = re.compile(r"^Reason:\s*(?P<reason>.+?)\s*$", re.MULTILINE)


class LoopError(RuntimeError):
    """Raised when the autoresearch loop cannot continue safely."""


@dataclass(frozen=True)
class ExperimentChoice:
    experiment_id: str
    hypothesis: str
    short_description: str


@dataclass(frozen=True)
class RewardTelemetrySnapshot:
    reward_mean: float
    reward_std: float
    gate_factor_mean: float
    compile_mean: float
    behavior_mean: float
    behavior_from_execution_mean: float
    json_valid_mean: float
    signature_mean: float


def tag_for_date(now: datetime) -> str:
    return now.strftime("%b%d").lower() + "-grpo"


def score_metrics(metrics: Mapping[str, float]) -> float:
    return (
        0.30 * float(metrics.get("behavior_success_rate", 0.0))
        + 0.25 * float(metrics.get("compile_success_rate", 0.0))
        + 0.20 * float(metrics.get("json_valid_rate", 0.0))
        + 0.15 * float(metrics.get("readability_score", 0.0))
        + 0.10 * float(metrics.get("naming_score", 0.0))
    )


def parse_program_control(program_text: str) -> tuple[str, str]:
    if STOP_SECTION not in program_text:
        return "continue", "missing stop section"
    stop_section = program_text.split(STOP_SECTION, maxsplit=1)[1]
    stop_section = stop_section.split("\n## ", maxsplit=1)[0]
    status_match = _PROGRAM_STATUS_RE.search(stop_section)
    reason_match = _PROGRAM_REASON_RE.search(stop_section)
    status = status_match.group("status").strip().lower() if status_match else "continue"
    reason = reason_match.group("reason").strip() if reason_match else "unspecified"
    return status, reason


def classify_invalid_prediction_rows(rows: list[dict[str, Any]]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("json_valid", True):
            continue
        raw_text = str(row.get("raw_text") or "")
        _, valid = normalize_output_with_status(raw_text)
        if valid:
            continue
        try:
            parsed = json.loads(raw_text)
        except Exception:  # noqa: BLE001
            reasons["json_parse_error"] += 1
            continue
        renamings = parsed.get("renamings")
        if not isinstance(renamings, dict):
            reasons[f"renamings_{type(renamings).__name__}"] += 1
            continue
        reasons["other_schema_error"] += 1
    return reasons


def choose_experiment(
    champion_metrics: Mapping[str, float],
    invalid_reasons: Mapping[str, int],
    prior_experiment_ids: set[str],
    *,
    recent_entries: list[Mapping[str, Any]] | None = None,
    reward_telemetry: RewardTelemetrySnapshot | None = None,
) -> ExperimentChoice:
    recent_entries = list(recent_entries or [])
    prompt_round = (
        sum(
            1
            for item in prior_experiment_ids
            if item.startswith("runtime_prompt_contract_v")
        )
        + 1
    )
    format_round = (
        sum(
            1 for item in prior_experiment_ids if item.startswith("reward_format_bias_v")
        )
        + 1
    )
    safety_round = (
        sum(1 for item in prior_experiment_ids if item.startswith("reward_safety_bias_v"))
        + 1
    )
    hard_safety_round = (
        sum(1 for item in prior_experiment_ids if item.startswith("reward_hard_safety_gate_v"))
        + 1
    )
    signature_round = (
        sum(1 for item in prior_experiment_ids if item.startswith("reward_signature_bias_v"))
        + 1
    )
    cooling_round = (
        sum(1 for item in prior_experiment_ids if item.startswith("rollout_cooling_v")) + 1
    )

    json_valid_rate = float(champion_metrics.get("json_valid_rate", 0.0))
    compile_rate = float(champion_metrics.get("compile_success_rate", 0.0))
    behavior_rate = float(champion_metrics.get("behavior_success_rate", 0.0))
    recent_metric_payloads = [
        entry.get("metrics")
        for entry in recent_entries[-4:]
        if isinstance(entry.get("metrics"), Mapping)
    ]
    recent_compile_floor = min(
        float(metrics.get("compile_success_rate", compile_rate))
        for metrics in recent_metric_payloads
    ) if recent_metric_payloads else compile_rate
    recent_behavior_floor = min(
        float(metrics.get("behavior_success_rate", behavior_rate))
        for metrics in recent_metric_payloads
    ) if recent_metric_payloads else behavior_rate
    recent_json_floor = min(
        float(metrics.get("json_valid_rate", json_valid_rate))
        for metrics in recent_metric_payloads
    ) if recent_metric_payloads else json_valid_rate
    reward_collapse = (
        reward_telemetry is not None
        and reward_telemetry.behavior_from_execution_mean >= 0.5
        and reward_telemetry.reward_mean >= 9.5
        and reward_telemetry.reward_std <= 1.0
        and reward_telemetry.gate_factor_mean >= 0.95
    )
    recent_safety_regression = (
        recent_compile_floor < compile_rate - 0.03
        or recent_behavior_floor < behavior_rate - 0.02
    )

    if (compile_rate < 0.65 or recent_safety_regression) and hard_safety_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_hard_safety_gate_v{hard_safety_round}",
            hypothesis=(
                "Making compile and behavior failures collapse reward much harder should reduce "
                "the reward hacking pattern where training reward rises while validation compile "
                "and behavior regress."
            ),
            short_description="reward hard safety gate",
        )

    if reward_collapse and cooling_round <= 3:
        return ExperimentChoice(
            experiment_id=f"rollout_cooling_v{cooling_round}",
            hypothesis=(
                "Cooling the GRPO rollout configuration should reduce low-variance reward "
                "collapse and improve validation generalization."
            ),
            short_description="rollout cooling",
        )

    if (compile_rate < 0.68 or behavior_rate < 0.88) and signature_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_signature_bias_v{signature_round}",
            hypothesis=(
                "Reweighting reward toward signature fidelity and away from cosmetic cleanup "
                "should preserve compile and behavior better than the current champion."
            ),
            short_description="reward signature bias",
        )

    if json_valid_rate < 0.95 and prompt_round <= 3:
        if invalid_reasons.get("json_parse_error", 0) >= invalid_reasons.get("renamings_list", 0):
            return ExperimentChoice(
                experiment_id=f"runtime_prompt_contract_v{prompt_round}",
                hypothesis=(
                    "Tightening the runtime GRPO prompt contract will reduce truncated JSON and "
                    "list-valued renamings without spending more completion tokens."
                ),
                short_description="runtime prompt contract",
            )
        return ExperimentChoice(
            experiment_id=f"runtime_prompt_contract_v{prompt_round}",
            hypothesis=(
                "Making the runtime GRPO prompt spell out the renamings object contract will "
                "reduce schema-invalid outputs."
            ),
            short_description="runtime prompt contract",
        )

    if json_valid_rate < 0.95 and format_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_format_bias_v{format_round}",
            hypothesis=(
                "Increasing the relative reward value of schema-valid completions will make JSON "
                "validity more stable than cleanup polish."
            ),
            short_description="reward format bias",
        )

    if (compile_rate < 0.65 or behavior_rate < 0.85) and safety_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_safety_bias_v{safety_round}",
            hypothesis=(
                "Increasing compile and behavior reward weight should preserve the champion's "
                "safety metrics better than format-only reward changes."
            ),
            short_description="reward safety bias",
        )

    if recent_json_floor < 0.95 and format_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_format_bias_v{format_round}",
            hypothesis=(
                "Increasing the relative reward value of schema-valid completions may recover "
                "JSON validity without reopening the reward-gating changes."
            ),
            short_description="reward format bias",
        )

    if prompt_round <= 3:
        return ExperimentChoice(
            experiment_id=f"runtime_prompt_contract_v{prompt_round}",
            hypothesis=(
                "A stricter runtime prompt contract is still the highest-leverage low-complexity "
                "change for this GRPO profile."
            ),
            short_description="runtime prompt contract",
        )
    if format_round <= 3:
        return ExperimentChoice(
            experiment_id=f"reward_format_bias_v{format_round}",
            hypothesis=(
                "A modest reward rebalance toward format reliability may improve the scalar score "
                "without expanding the experiment surface."
            ),
            short_description="reward format bias",
        )
    return ExperimentChoice(
        experiment_id=f"reward_hard_safety_gate_v{hard_safety_round}",
        hypothesis=(
            "A harder compile and behavior failure gate is the next bounded fallback after the "
            "prompt, format, and signature-focused experiments."
        ),
        short_description="reward hard safety gate",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_baseline(path: Path, payload: Mapping[str, Any]) -> None:
    baseline_payload = {key: payload.get(key) for key in _BASELINE_KEYS if key in payload}
    path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8")


def _repo_root(start: Path | None = None) -> Path:
    return ProjectPaths.discover(start=start)


def _run_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted(
        (root / "artifacts" / "runs").glob(f"{prefix}-*"),
        key=lambda path: path.stat().st_mtime,
    )


def _latest_run_dir(root: Path, prefix: str) -> Path | None:
    candidates = _run_dirs(root, prefix)
    return candidates[-1] if candidates else None


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def _git_branch_names(root: Path) -> list[str]:
    result = _git(root, "branch", "--format=%(refname:short)")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _current_branch(root: Path) -> str:
    return _git(root, "branch", "--show-current").stdout.strip()


def _has_clean_worktree(root: Path) -> bool:
    return not _git(root, "status", "--short").stdout.strip()


def _dirty_paths(root: Path) -> list[str]:
    status = _git(root, "status", "--short", "--untracked-files=all").stdout.splitlines()
    paths: list[str] = []
    for line in status:
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", maxsplit=1)[1].strip()
        paths.append(path.replace("\\", "/"))
    return paths


def _path_matches_surface(path: str, surface: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for candidate in surface:
        prefix = candidate.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    return False


def _conflicting_dirty_paths(root: Path) -> list[str]:
    if _has_clean_worktree(root):
        return []
    dirty = _dirty_paths(root)
    return [
        path
        for path in dirty
        if _path_matches_surface(path, EXPERIMENT_SURFACE)
        and not any(
            path.startswith(prefix) or path == prefix.rstrip("/")
            for prefix in SAFE_DIRTY_PREFIXES
        )
    ]


def _worktree_is_safe(root: Path) -> bool:
    return not _conflicting_dirty_paths(root)


def _git_switch(
    root: Path,
    branch: str,
    *,
    create: bool = False,
    start_point: str | None = None,
) -> None:
    args = ["switch"]
    if create:
        args.extend(["-c", branch])
        if start_point:
            args.append(start_point)
    else:
        args.append(branch)
    _git(root, *args)


def _git_commit_paths(root: Path, message: str, paths: list[str]) -> str:
    _git(root, "add", "--", *paths)
    cached = _git(root, "diff", "--cached", "--name-only", "--", *paths)
    if not cached.stdout.strip():
        raise LoopError("candidate experiment produced no staged changes")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _git_latest_touch_timestamp(root: Path, paths: tuple[str, ...]) -> int:
    result = _git(root, "log", "-1", "--format=%ct", "--", *paths, check=False)
    value = result.stdout.strip()
    return int(value) if value else 0


def _run_logged(
    root: Path,
    logger: logging.Logger,
    args: list[str],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    logger.info("running command: %s", " ".join(args))
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    process = subprocess.Popen(
        args,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
        creationflags=creationflags,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info(line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        raise LoopError(f"command failed with exit code {returncode}: {' '.join(args)}")


def _new_run_from_command(root: Path, prefix: str, args: list[str], logger: logging.Logger) -> Path:
    before = {path.name for path in _run_dirs(root, prefix)}
    _run_logged(root, logger, args)
    after = [path for path in _run_dirs(root, prefix) if path.name not in before]
    if not after:
        raise LoopError(f"command did not create a new {prefix} run")
    return after[-1]


def _training_profile_path(root: Path, profile: str) -> Path:
    return root / "configs" / "training" / f"{profile}.yaml"


def _config_snapshot(root: Path, profile: str) -> dict[str, Any]:
    profile_text = _training_profile_path(root, profile).read_text(encoding="utf-8")
    profile_payload = yaml.safe_load(profile_text)
    training = profile_payload.get("training", {})
    return {
        "learning_rate": training.get("learning_rate"),
        "max_completion_length": training.get("max_completion_length"),
        "max_prompt_length": training.get("max_prompt_length"),
        "generations_per_prompt": training.get("generations_per_prompt"),
        "warmup_ratio": training.get("warmup_ratio"),
        "behavior_similarity_threshold": training.get("behavior_similarity_threshold"),
        "execution_pass_rate_threshold": training.get("execution_pass_rate_threshold"),
        "min_completion_ratio": training.get("min_completion_ratio"),
        "max_grad_norm": training.get("max_grad_norm"),
        "reward_weights": training.get("reward_weights", {}),
    }


def _metrics_from_eval_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    metrics = manifest.get("metrics", {})
    return {key: float(metrics.get(key, 0.0)) for key in _SCORE_KEYS}


def _hard_keep_gates_pass(metrics: Mapping[str, float], champion: Mapping[str, float]) -> bool:
    return (
        float(metrics.get("compile_success_rate", 0.0))
        >= float(champion.get("compile_success_rate", 0.0)) - 0.01
        and float(metrics.get("behavior_success_rate", 0.0))
        >= float(champion.get("behavior_success_rate", 0.0)) - 0.01
        and float(metrics.get("json_valid_rate", 0.0))
        >= float(champion.get("json_valid_rate", 0.0)) - 0.02
    )


def _baseline_from_existing_run(
    root: Path,
    profile: str,
) -> tuple[dict[str, Any], Path, Path] | None:
    latest_eval_dir = _latest_run_dir(root, "eval-grpo-checkpoint")
    if latest_eval_dir is None:
        return None
    eval_manifest_path = latest_eval_dir / "checkpoint_eval_manifest.json"
    if not eval_manifest_path.exists():
        return None
    eval_manifest = _load_json(eval_manifest_path)
    if eval_manifest.get("training_profile") != profile or eval_manifest.get("split") != "val":
        return None
    checkpoint_dir = Path(str(eval_manifest.get("checkpoint_dir", "")))
    train_manifest_path = checkpoint_dir / "grpo_training_manifest.json"
    if not train_manifest_path.exists():
        return None
    consistency_ts = _git_latest_touch_timestamp(root, CONSISTENCY_SURFACE)
    eval_ts = int(eval_manifest_path.stat().st_mtime)
    if consistency_ts > eval_ts:
        return None
    metrics = _metrics_from_eval_manifest(eval_manifest)
    payload = {
        "score": score_metrics(metrics),
        **metrics,
        "iteration": 0,
        "train_run_id": checkpoint_dir.parent.name,
        "eval_run_id": latest_eval_dir.name,
        "timestamp": datetime.now().astimezone().isoformat(),
        "config_snapshot": _config_snapshot(root, profile),
    }
    return payload, checkpoint_dir.parent, latest_eval_dir


def _validate_prerequisites(root: Path, profile: str, logger: logging.Logger) -> None:
    if sys.platform != "win32":
        raise LoopError("GRPO autoresearch requires Windows")
    dataset_path = root / "data" / "processed" / "rl" / "rl_records.jsonl"
    if not dataset_path.exists():
        raise LoopError("missing RL dataset at data/processed/rl/rl_records.jsonl")
    if not list((root / "artifacts" / "runs").glob("train-sft-*")):
        raise LoopError("missing SFT checkpoint run; train-grpo cannot resolve a base model")
    for command in (
        [sys.executable, "-m", "decomp_clarifier.cli", "train-grpo", "--help"],
        [sys.executable, "-m", "decomp_clarifier.cli", "eval-grpo-checkpoint", "--help"],
        [
            sys.executable,
            "-c",
            "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)",
        ],
    ):
        logger.info("preflight: %s", " ".join(command))
        subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    logger.info("validated prerequisites for training profile %s", profile)


def _load_log_entries(log_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(log_path)


def _next_iteration(entries: list[dict[str, Any]]) -> int:
    if not entries:
        return 0
    return max(int(entry.get("iteration", 0)) for entry in entries) + 1


def _load_predictions_invalid_reasons(eval_dir: Path) -> Counter[str]:
    predictions_path = eval_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return Counter()
    return classify_invalid_prediction_rows(_read_jsonl(predictions_path))


def _reward_telemetry_from_train_dir(train_dir: Path) -> RewardTelemetrySnapshot | None:
    metrics_path = train_dir / "model" / "logs" / "grpo_metrics.jsonl"
    if not metrics_path.exists():
        return None
    for row in reversed(_read_jsonl(metrics_path)):
        if row.get("source") != "reward_func":
            continue
        return RewardTelemetrySnapshot(
            reward_mean=float(row.get("reward_mean", 0.0)),
            reward_std=float(row.get("reward_std", 0.0)),
            gate_factor_mean=float(row.get("components/gate_factor_mean", 0.0)),
            compile_mean=float(row.get("components/compile_mean", 0.0)),
            behavior_mean=float(row.get("components/behavior_mean", 0.0)),
            behavior_from_execution_mean=float(
                row.get("components/behavior_from_execution_mean", 0.0)
            ),
            json_valid_mean=float(row.get("components/json_valid_mean", 0.0)),
            signature_mean=float(row.get("components/signature_mean", 0.0)),
        )
    return None


def _latest_reward_telemetry(
    root: Path,
    entries: list[Mapping[str, Any]],
) -> RewardTelemetrySnapshot | None:
    seen_run_ids: set[str] = set()
    for entry in reversed(entries):
        run_id = str(entry.get("train_run_id") or "")
        if not run_id or run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        snapshot = _reward_telemetry_from_train_dir(root / "artifacts" / "runs" / run_id)
        if snapshot is not None:
            return snapshot
    latest_train_dir = _latest_run_dir(root, "train-grpo")
    if latest_train_dir is None:
        return None
    return _reward_telemetry_from_train_dir(latest_train_dir)


def _render_runtime_prompt_contract(variant: int) -> str:
    extra_lines = {
        1: [
            'Use a JSON object for renamings, for example {"param_1":"cfg"}; '
            "use {} when no safe rename exists.",
            "Keep summary to one short sentence.",
            "Keep cleaned_c to exactly one concise reconstructed function.",
        ],
        2: [
            'Use a JSON object for renamings, for example {"param_1":"cfg"}; '
            "never emit a list.",
            "If you are unsure about a rename, leave it out and return {} instead of guessing.",
            'Output shape: {"summary":"...","confidence":0.0,'
            '"renamings":{"old":"new"},"cleaned_c":"int f(...) { ... }"}',
            "Keep cleaned_c to exactly one concise reconstructed function.",
        ],
        3: [
            "Use a JSON object for renamings, never a list; return {} "
            "when there are no safe renames.",
            "Summary must be one short sentence with no list formatting.",
            "cleaned_c must contain exactly one function and should be as "
            "short as possible while staying valid.",
            "Do not add helper functions, comments, or project-wide rewrites.",
        ],
    }[variant]
    joined_extra = "\n            ".join(f"{line!r}," for line in extra_lines)
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import re\n"
        "from pathlib import Path\n"
        "from typing import Any\n\n\n"
        "_PROMPT_SECTION_RE = re.compile(\n"
        '    r"Task:\\s*(?P<task>[^\\n]+).*?Decompiler:\\n(?P<decompiler>.*?)\\n\\n"\n'
        '    r"Imports:\\s*(?P<imports>.*?)\\nCallees:\\s*(?P<callees>.*?)\\n"\n'
        '    r"Semantic summary:\\s*(?P<summary>.*?)\\nJSON:\\s*$",\n'
        "    re.DOTALL,\n"
        ")\n\n\n"
        "def load_rl_records(path: Path) -> list[dict[str, Any]]:\n"
        "    rows: list[dict[str, Any]] = []\n"
        "    for line in path.read_text(encoding=\"utf-8\").splitlines():\n"
        "        if line.strip():\n"
        "            rows.append(json.loads(line))\n"
        "    return rows\n\n\n"
        "def prompt_from_record(record: dict[str, Any]) -> str:\n"
        "    prompt = str(record.get(\"prompt\", \"\"))\n"
        "    normalized = prompt.replace(\"\\r\\n\", \"\\n\")\n"
        "    match = _PROMPT_SECTION_RE.search(normalized)\n"
        "    if match is None:\n"
        "        return prompt\n"
        "    task_type = str(record.get(\"task_type\") or match.group(\"task\").strip())\n"
        "    task_focus = {\n"
        '        "rename": "Prefer conservative renames and return {} when unsure.",\n'
        '        "cleanup": "Simplify the current function without inventing new helpers.",\n'
        '        "full_clarify": "Recover one readable function from the '
        'binary-grounded evidence.",\n'
        '    }.get(task_type, "Keep the output concise and schema-valid.")\n'
        "    return \"\\n\".join(\n"
        "        [\n"
        '            "You are a binary-grounded code clarification assistant.",\n'
        '            f"Task: {task_type}",\n'
        '            "Return exactly one JSON object with keys summary, '
        'confidence, renamings, cleaned_c.",\n'
        f"            {joined_extra}\n"
        '            f"Task focus: {task_focus}",\n'
        '            "Do not include markdown, commentary, XML tags, or <think> blocks.",\n'
        '            "",\n'
        '            "Decompiler:",\n'
        '            match.group("decompiler").strip(),\n'
        '            "",\n'
        '            f"Imports: {match.group(\'imports\').strip()}",\n'
        '            f"Callees: {match.group(\'callees\').strip()}",\n'
        '            f"Semantic summary: {match.group(\'summary\').strip()}",\n'
        '            "JSON:",\n'
        "        ]\n"
        "    )\n\n\n"
        "def reward_fields_from_record(record: dict[str, Any]) -> dict[str, Any]:\n"
        "    return {\n"
        '        "task_type": record.get("task_type", "full_clarify"),\n'
        '        "source_function_name": record.get("source_function_name", ""),\n'
        '        "raw_code": record.get("raw_code", ""),\n'
        '        "compile_reference_source": record.get(\n'
        '            "compile_reference_source", record.get("target_clean_code", "")\n'
        "        ),\n"
        '        "target_clean_code": record.get("target_clean_code", ""),\n'
        '        "target_renamings": record.get("target_renamings", "{}"),\n'
        '        "allowed_imports": record.get("allowed_imports", "[]"),\n'
        '        "allowed_callees": record.get("allowed_callees", "[]"),\n'
        '        "compiler_executable": record.get("compiler_executable"),\n'
        '        "tests_ref": record.get("tests_ref") or "",\n'
        "    }\n"
    )


def _apply_runtime_prompt_contract(root: Path, variant: int) -> list[str]:
    path = root / "src" / "decomp_clarifier" / "training" / "grpo" / "data.py"
    path.write_text(_render_runtime_prompt_contract(variant), encoding="utf-8")
    return ["src/decomp_clarifier/training/grpo/data.py"]


def _apply_reward_format_bias(root: Path, variant: int) -> list[str]:
    path = root / "configs" / "training" / "grpo_qwen35_2b.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = payload.setdefault("training", {}).setdefault("reward_weights", {})
    weights["format"] = round(1.0 + 0.2 * variant, 2)
    weights["cleanup"] = round(max(1.0, 1.5 - 0.1 * variant), 2)
    weights["readability"] = round(max(0.7, 1.0 - 0.1 * variant), 2)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return ["configs/training/grpo_qwen35_2b.yaml"]


def _apply_reward_safety_bias(root: Path, variant: int) -> list[str]:
    path = root / "configs" / "training" / "grpo_qwen35_2b.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = payload.setdefault("training", {}).setdefault("reward_weights", {})
    weights["compile"] = round(3.0 + 0.25 * variant, 2)
    weights["behavior"] = round(2.0 + 0.25 * variant, 2)
    weights["cleanup"] = round(max(1.0, 1.5 - 0.1 * variant), 2)
    weights["readability"] = round(max(0.8, 1.0 - 0.05 * variant), 2)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return ["configs/training/grpo_qwen35_2b.yaml"]


def _replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise LoopError(f"expected text not found while applying experiment: {old!r}")
    return text.replace(old, new)


def _apply_reward_hard_safety_gate(root: Path, variant: int) -> list[str]:
    path = root / "src" / "decomp_clarifier" / "training" / "grpo" / "rewards.py"
    text = path.read_text(encoding="utf-8")
    compile_gate = {1: "0.0", 2: "0.0", 3: "0.0"}[variant]
    behavior_gate = {1: "0.0", 2: "0.0", 3: "0.0"}[variant]
    compile_penalty = {1: "0.25", 2: "0.5", 3: "0.75"}[variant]
    behavior_penalty = {1: "0.15", 2: "0.25", 3: "0.35"}[variant]
    replacements = {
        "_COMPILE_FAILURE_GATE": compile_gate,
        "_BEHAVIOR_FAILURE_GATE": behavior_gate,
        "_COMPILE_FAILURE_PENALTY": compile_penalty,
        "_BEHAVIOR_FAILURE_PENALTY": behavior_penalty,
    }
    for constant, value in replacements.items():
        updated, count = re.subn(
            rf"{constant} = [0-9.]+",
            f"{constant} = {value}",
            text,
        )
        if count != 1:
            raise LoopError(f"failed to update {constant} in rewards.py")
        text = updated
    path.write_text(text, encoding="utf-8")
    return ["src/decomp_clarifier/training/grpo/rewards.py"]


def _apply_reward_signature_bias(root: Path, variant: int) -> list[str]:
    path = root / "configs" / "training" / "grpo_qwen35_2b.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = payload.setdefault("training", {}).setdefault("reward_weights", {})
    weights["signature"] = round(1.5 + 0.5 * variant, 2)
    weights["cleanup"] = round(max(0.8, 1.4 - 0.2 * variant), 2)
    weights["readability"] = round(max(0.7, 0.95 - 0.1 * variant), 2)
    weights["hallucination_penalty"] = round(2.0 + 0.25 * variant, 2)
    weights["decompiler_type_penalty"] = round(1.0 + 0.25 * variant, 2)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return ["configs/training/grpo_qwen35_2b.yaml"]


def _apply_rollout_cooling(root: Path, variant: int) -> list[str]:
    path = root / "configs" / "training" / "grpo_qwen35_2b.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = payload.setdefault("training", {})
    learning_rates = {1: 1.5e-06, 2: 1.0e-06, 3: 7.5e-07}
    generations = {1: 3, 2: 2, 3: 2}
    completion_lengths = {1: 320, 2: 288, 3: 256}
    training["learning_rate"] = learning_rates[variant]
    training["warmup_ratio"] = 0.05
    training["generations_per_prompt"] = generations[variant]
    training["max_completion_length"] = completion_lengths[variant]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return ["configs/training/grpo_qwen35_2b.yaml"]


def _apply_experiment(root: Path, choice: ExperimentChoice) -> list[str]:
    if choice.experiment_id.startswith("runtime_prompt_contract_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_runtime_prompt_contract(root, variant)
    if choice.experiment_id.startswith("reward_format_bias_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_reward_format_bias(root, variant)
    if choice.experiment_id.startswith("reward_safety_bias_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_reward_safety_bias(root, variant)
    if choice.experiment_id.startswith("reward_hard_safety_gate_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_reward_hard_safety_gate(root, variant)
    if choice.experiment_id.startswith("reward_signature_bias_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_reward_signature_bias(root, variant)
    if choice.experiment_id.startswith("rollout_cooling_v"):
        variant = int(choice.experiment_id.rsplit("v", maxsplit=1)[1])
        return _apply_rollout_cooling(root, variant)
    raise LoopError(f"unsupported experiment id {choice.experiment_id}")


def _run_targeted_tests(root: Path, logger: logging.Logger) -> None:
    _run_logged(
        root,
        logger,
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_ghidra_dataset_eval.py",
            "tests/unit/test_training.py",
            "tests/unit/test_autoresearch.py",
            "-q",
        ],
    )


def _record_payload(
    *,
    iteration: int,
    tag: str,
    champion_branch: str,
    candidate_branch: str,
    parent_commit: str,
    candidate_commit: str,
    status: str,
    hypothesis: str,
    experiment_id: str,
    files_touched: list[str],
    train_run_id: str | None,
    scout_eval_run_id: str | None,
    confirm_eval_run_id: str | None,
    scout_score: float | None,
    confirm_score: float | None,
    metrics: Mapping[str, float],
    notes: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "timestamp": datetime.now().astimezone().isoformat(),
        "tag": tag,
        "champion_branch": champion_branch,
        "candidate_branch": candidate_branch,
        "parent_commit": parent_commit,
        "candidate_commit": candidate_commit,
        "status": status,
        "hypothesis": hypothesis,
        "experiment_id": experiment_id,
        "files_touched": files_touched,
        "train_run_id": train_run_id,
        "scout_eval_run_id": scout_eval_run_id,
        "confirm_eval_run_id": confirm_eval_run_id,
        "scout_score": scout_score,
        "confirm_score": confirm_score,
        "metrics": {key: float(metrics.get(key, 0.0)) for key in _SCORE_KEYS},
        "notes": notes,
    }


def _next_candidate_branch(
    tag: str,
    entries: list[Mapping[str, Any]],
    existing_branches: list[str],
) -> tuple[int, str]:
    iteration = _next_iteration(entries)
    branch_names = set(existing_branches)
    while True:
        candidate_branch = f"autoresearch-tmp/{tag}-{iteration:04d}"
        if candidate_branch not in branch_names:
            return iteration, candidate_branch
        iteration += 1


class AutoresearchLoop:
    def __init__(
        self,
        *,
        root: Path,
        tag: str | None,
        training_profile: str,
        logger: logging.Logger,
    ) -> None:
        self.root = root
        self.tag = tag or tag_for_date(datetime.now().astimezone())
        self.training_profile = training_profile
        self.logger = logger
        self.program_path = root / "research" / "program.md"
        self.baseline_path = root / "research" / "baseline.json"
        self.log_path = root / "research" / "experiment_log.jsonl"
        self.champion_branch = f"autoresearch/{self.tag}"

    def run(self) -> None:
        _validate_prerequisites(self.root, self.training_profile, self.logger)
        dirty_conflicts = _conflicting_dirty_paths(self.root)
        if dirty_conflicts:
            raise LoopError(
                "working tree contains dirty experiment-surface paths: "
                + ", ".join(dirty_conflicts)
            )
        self._bootstrap_baseline()
        while True:
            status, reason = parse_program_control(self.program_path.read_text(encoding="utf-8"))
            if status == "stop":
                self.logger.info("stop requested by research/program.md: %s", reason)
                return
            self._run_iteration()

    def _bootstrap_baseline(self) -> None:
        branches = _git_branch_names(self.root)
        if self.champion_branch not in branches:
            start_point = _current_branch(self.root)
            _git_switch(self.root, self.champion_branch, create=True, start_point=start_point)
            self.logger.info(
                "created champion branch %s from %s",
                self.champion_branch,
                start_point,
            )
        else:
            _git_switch(self.root, self.champion_branch)
            self.logger.info("resuming champion branch %s", self.champion_branch)

        entries = _load_log_entries(self.log_path)
        if entries:
            return

        adopted = _baseline_from_existing_run(self.root, self.training_profile)
        if adopted is not None:
            baseline_payload, train_dir, eval_dir = adopted
            record = _record_payload(
                iteration=0,
                tag=self.tag,
                champion_branch=self.champion_branch,
                candidate_branch=self.champion_branch,
                parent_commit=_git(self.root, "rev-parse", "HEAD").stdout.strip(),
                candidate_commit=_git(self.root, "rev-parse", "HEAD").stdout.strip(),
                status="baseline",
                hypothesis=(
                    "Bootstrap the champion from the latest comparable GRPO "
                    "checkpoint with no code changes."
                ),
                experiment_id="baseline_bootstrap_existing_run",
                files_touched=[],
                train_run_id=train_dir.name,
                scout_eval_run_id=eval_dir.name,
                confirm_eval_run_id=eval_dir.name,
                scout_score=float(baseline_payload["score"]),
                confirm_score=float(baseline_payload["score"]),
                metrics={key: float(baseline_payload[key]) for key in _SCORE_KEYS},
                notes=(
                    "Adopted the latest GRPO run as baseline because the evaluation and experiment "
                    "surface were unchanged after that run."
                ),
            )
            _write_baseline(self.baseline_path, baseline_payload)
            _append_jsonl(self.log_path, record)
            self.logger.info(
                "baseline adopted from existing run train=%s eval=%s score=%.4f",
                train_dir.name,
                eval_dir.name,
                float(baseline_payload["score"]),
            )
            return

        self.logger.info(
            "no reusable baseline found; running baseline training from "
            "champion branch"
        )
        parent_commit = _git(self.root, "rev-parse", "HEAD").stdout.strip()
        train_dir = _new_run_from_command(
            self.root,
            "train-grpo",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "train-grpo",
                "--training-profile",
                self.training_profile,
            ],
            self.logger,
        )
        scout_dir = _new_run_from_command(
            self.root,
            "eval-grpo-checkpoint",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "eval-grpo-checkpoint",
                "--training-profile",
                self.training_profile,
                "--split",
                "val",
                "--sample-limit",
                "50",
            ],
            self.logger,
        )
        confirm_dir = _new_run_from_command(
            self.root,
            "eval-grpo-checkpoint",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "eval-grpo-checkpoint",
                "--training-profile",
                self.training_profile,
                "--split",
                "val",
            ],
            self.logger,
        )
        confirm_manifest = _load_json(confirm_dir / "checkpoint_eval_manifest.json")
        metrics = _metrics_from_eval_manifest(confirm_manifest)
        baseline_score = score_metrics(metrics)
        baseline_payload = {
            "score": baseline_score,
            **metrics,
            "iteration": 0,
            "train_run_id": train_dir.name,
            "eval_run_id": confirm_dir.name,
            "timestamp": datetime.now().astimezone().isoformat(),
            "config_snapshot": _config_snapshot(self.root, self.training_profile),
        }
        record = _record_payload(
            iteration=0,
            tag=self.tag,
            champion_branch=self.champion_branch,
            candidate_branch=self.champion_branch,
            parent_commit=parent_commit,
            candidate_commit=parent_commit,
            status="baseline",
            hypothesis="Establish the first GRPO champion with no code changes.",
            experiment_id="baseline_bootstrap_train",
            files_touched=[],
            train_run_id=train_dir.name,
            scout_eval_run_id=scout_dir.name,
            confirm_eval_run_id=confirm_dir.name,
            scout_score=score_metrics(
                _metrics_from_eval_manifest(
                    _load_json(scout_dir / "checkpoint_eval_manifest.json")
                )
            ),
            confirm_score=baseline_score,
            metrics=metrics,
            notes="Baseline established by training and evaluating the current champion branch.",
        )
        _write_baseline(self.baseline_path, baseline_payload)
        _append_jsonl(self.log_path, record)

    def _run_iteration(self) -> None:
        entries = _load_log_entries(self.log_path)
        baseline = _load_json(self.baseline_path)
        champion_metrics = {key: float(baseline.get(key, 0.0)) for key in _SCORE_KEYS}
        latest_eval_dir = _latest_run_dir(self.root, "eval-grpo-checkpoint")
        invalid_reasons = (
            _load_predictions_invalid_reasons(latest_eval_dir)
            if latest_eval_dir is not None
            else Counter()
        )
        reward_telemetry = _latest_reward_telemetry(self.root, entries)
        choice = choose_experiment(
            champion_metrics,
            invalid_reasons,
            {str(entry.get("experiment_id", "")) for entry in entries},
            recent_entries=entries,
            reward_telemetry=reward_telemetry,
        )
        iteration, candidate_branch = _next_candidate_branch(
            self.tag,
            entries,
            _git_branch_names(self.root),
        )
        _git_switch(self.root, self.champion_branch)
        parent_commit = _git(self.root, "rev-parse", "HEAD").stdout.strip()
        _git_switch(self.root, candidate_branch, create=True)
        self.logger.info(
            "iteration=%s branch=%s hypothesis=%s",
            iteration,
            candidate_branch,
            choice.hypothesis,
        )

        files_touched: list[str] = []
        candidate_commit = parent_commit
        train_dir: Path | None = None
        scout_dir: Path | None = None
        confirm_dir: Path | None = None
        scout_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
        confirm_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
        scout_score: float | None = None
        confirm_score: float | None = None
        status = "discard"
        notes = ""

        try:
            files_touched = _apply_experiment(self.root, choice)
            self.logger.info("touched files: %s", ", ".join(files_touched))
            _run_targeted_tests(self.root, self.logger)
            candidate_commit = _git_commit_paths(
                self.root,
                f"research: try {choice.short_description}",
                files_touched,
            )
            train_dir = _new_run_from_command(
                self.root,
                "train-grpo",
                [
                    sys.executable,
                    "-m",
                    "decomp_clarifier.cli",
                    "train-grpo",
                    "--training-profile",
                    self.training_profile,
                ],
                self.logger,
            )
            scout_dir = _new_run_from_command(
                self.root,
                "eval-grpo-checkpoint",
                [
                    sys.executable,
                    "-m",
                    "decomp_clarifier.cli",
                    "eval-grpo-checkpoint",
                    "--training-profile",
                    self.training_profile,
                    "--split",
                    "val",
                    "--sample-limit",
                    "50",
                ],
                self.logger,
            )
            scout_metrics = _metrics_from_eval_manifest(
                _load_json(scout_dir / "checkpoint_eval_manifest.json")
            )
            scout_score = score_metrics(scout_metrics)
            champion_score = float(baseline.get("score", 0.0))
            if not _hard_keep_gates_pass(scout_metrics, champion_metrics):
                status = "discard"
                notes = "Scout evaluation failed a hard keep gate against the champion."
            elif scout_score < champion_score + 0.005:
                status = "discard"
                notes = "Scout score did not clear the +0.005 confirm threshold."
            else:
                confirm_dir = _new_run_from_command(
                    self.root,
                    "eval-grpo-checkpoint",
                    [
                        sys.executable,
                        "-m",
                        "decomp_clarifier.cli",
                        "eval-grpo-checkpoint",
                        "--training-profile",
                        self.training_profile,
                        "--split",
                        "val",
                    ],
                    self.logger,
                )
                confirm_metrics = _metrics_from_eval_manifest(
                    _load_json(confirm_dir / "checkpoint_eval_manifest.json")
                )
                confirm_score = score_metrics(confirm_metrics)
                if (
                    _hard_keep_gates_pass(confirm_metrics, champion_metrics)
                    and confirm_score >= champion_score + 0.01
                ):
                    status = "keep"
                    notes = (
                        "Confirm evaluation cleared the keep gates and beat "
                        "the champion by at least 0.01."
                    )
                else:
                    status = "discard"
                    notes = "Confirm evaluation failed the keep threshold or a hard gate."
        except Exception as exc:  # noqa: BLE001
            status = "crash"
            notes = str(exc)
            self.logger.exception("candidate crashed: %s", exc)
        finally:
            if files_touched and _current_branch(self.root) == candidate_branch:
                _git(
                    self.root,
                    "restore",
                    "--source=HEAD",
                    "--staged",
                    "--worktree",
                    "--",
                    *files_touched,
                    check=False,
                )
            if _current_branch(self.root) != self.champion_branch:
                _git_switch(self.root, self.champion_branch)
            if status == "keep":
                _git(self.root, "merge", "--ff-only", candidate_branch)
                baseline_payload = {
                    "score": float(confirm_score or 0.0),
                    **confirm_metrics,
                    "iteration": iteration,
                    "train_run_id": train_dir.name if train_dir else None,
                    "eval_run_id": confirm_dir.name if confirm_dir else None,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "config_snapshot": _config_snapshot(self.root, self.training_profile),
                }
                _write_baseline(self.baseline_path, baseline_payload)
            record = _record_payload(
                iteration=iteration,
                tag=self.tag,
                champion_branch=self.champion_branch,
                candidate_branch=candidate_branch,
                parent_commit=parent_commit,
                candidate_commit=candidate_commit,
                status=status,
                hypothesis=choice.hypothesis,
                experiment_id=choice.experiment_id,
                files_touched=files_touched,
                train_run_id=train_dir.name if train_dir else None,
                scout_eval_run_id=scout_dir.name if scout_dir else None,
                confirm_eval_run_id=confirm_dir.name if confirm_dir else None,
                scout_score=scout_score,
                confirm_score=confirm_score,
                metrics=confirm_metrics if confirm_dir is not None else scout_metrics,
                notes=notes,
            )
            _append_jsonl(self.log_path, record)
            if candidate_branch in _git_branch_names(self.root):
                _git(self.root, "branch", "-D", candidate_branch)
            self.logger.info(
                "iteration=%s completed status=%s scout=%.4f confirm=%s",
                iteration,
                status,
                scout_score or 0.0,
                f"{confirm_score:.4f}" if confirm_score is not None else "n/a",
            )


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("decomp_clarifier.autoresearch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GRPO autoresearch loop.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--training-profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)

    root = _repo_root(args.root)
    tag = args.tag or tag_for_date(datetime.now().astimezone())
    log_file = args.log_file or (root / "artifacts" / "logs" / f"autoresearch-{tag}.log")
    logger = build_logger(log_file)
    try:
        loop = AutoresearchLoop(
            root=root,
            tag=tag,
            training_profile=args.training_profile,
            logger=logger,
        )
        loop.run()
    except LoopError as exc:
        logger.error("autoresearch stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
