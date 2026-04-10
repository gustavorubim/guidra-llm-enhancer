
# spec.md

# Binary-Grounded Decompiler Clarification with Qwen3.5, Unsloth, and GRPO

## 1. Purpose

Build a local research prototype that demonstrates the following claim:

> Starting from confusing Ghidra decompiler output, we can fine-tune an open model to produce code that is materially more digestible for a human reviewer.

This is **not** a perfect binary-to-source translator. It is a **decompiler clarification system**. The system should transform confusing, expert-heavy decompiled output into code that is easier to read, reason about, and review.

The project should prioritize:
- readability
- semantic plausibility
- better naming
- clearer structure
- safer human review experience

The project does **not** need to:
- recover original source exactly
- support many compilers or many architectures in v1
- support large real-world codebases in v1
- solve stripped-binary source recovery in the general case

---

## 2. Product framing

### 2.1 Core framing
We are building a **binary-grounded post-processor for decompiler output**, not an “inverse compiler.”

### 2.2 What success looks like
Given:
- Ghidra decompiled C
- assembly
- strings/imports
- light call-context metadata

The model produces:
- cleaner C-like code
- improved function and variable names
- simplified control flow where justified
- a short natural-language summary
- optional confidence notes and rename map

### 2.3 Why this framing matters
This framing is more aligned with what can actually be supervised:
- we can compare against the original synthetic source
- we can verify compilation and tests
- we can score readability and naming quality
- we do not need to claim exact source recovery

---

## 3. Goals and non-goals

## 3.1 Goals
1. Create a synthetic data factory:
   - generate readable C projects via OpenRouter
   - compile them
   - decompile them with Ghidra
   - assemble aligned training samples

2. Train a first model with SFT:
   - base model: **Qwen3.5 4B family**
   - framework: **Unsloth**
   - hardware target: **Windows + NVIDIA CUDA**
   - local environment: **project-local `.venv`**

3. Add RL after SFT:
   - algorithm family: **GRPO** via TRL/Unsloth-compatible workflow
   - reward functions focused on readability and faithfulness

4. Keep non-training tooling cross-platform:
   - macOS and Windows should both run:
     - sample generation
     - compilation
     - Ghidra extraction
     - dataset assembly
     - baseline inference
     - evaluation
     - reporting

5. Enforce engineering discipline:
   - modular repo
   - config-driven pipeline
   - at least **90% test coverage**
   - strong typing and linting
   - deterministic manifests and caching

## 3.2 Non-goals
1. Multi-compiler generalization in v1
2. Multi-architecture generalization in v1
3. Real malware support in v1
4. Exact reconstruction of original names in all cases
5. Full project-scale recompilation of recovered code for arbitrary binaries
6. Native local training support on macOS

---

## 4. Scope assumptions for v1

The user’s stated priority is demonstration, not universal coverage. Therefore v1 intentionally narrows scope.

### 4.1 Canonical simplifications
- One primary language: **C**
- One primary architecture family: **host-native x86_64 or arm64 where supported by the host compiler and Ghidra**
- One primary optimization regime: **mostly `-O0`, optional light challenge split at `-O2`**
- One primary compiler family: **Clang preferred**
- Small synthetic projects:
  - 1 to 8 source files
  - 3 to 20 functions per project
  - low to moderate use of pointers, structs, switches, loops, and helper functions
- Training objective centered on **digestibility**, not source exactness

### 4.2 Why mostly `-O0`
For a first demonstration:
- Ghidra output is still nontrivial
- source alignment is much cleaner
- RL rewards are easier to compute
- model progress is easier to measure

Later we can add a small curated `-O2` eval split.

---

## 5. High-level architecture

```text
OpenRouter synthetic code generation
        |
        v
Compile + test + metadata capture
        |
        v
Ghidra headless analysis + export
        |
        v
Dataset builder
  - align source, binary, decompiled output, assembly, metadata
  - derive training and evaluation splits
        |
        +--> baseline evaluation
        |
        +--> SFT dataset
        |
        +--> RL prompt/eval dataset
                    |
                    v
            Qwen3.5 + Unsloth SFT
                    |
                    v
         verifier and reward stack
                    |
                    v
              GRPO refinement
                    |
                    v
          evaluation + reporting + demo CLI
```

---

## 6. Platform support matrix

| Capability | macOS | Windows (CPU) | Windows + NVIDIA CUDA |
|---|---:|---:|---:|
| OpenRouter sample generation | Yes | Yes | Yes |
| Source normalization | Yes | Yes | Yes |
| Local compilation | Yes | Yes | Yes |
| Ghidra headless extraction | Yes | Yes | Yes |
| Dataset assembly | Yes | Yes | Yes |
| Baseline inference | Yes | Yes | Yes |
| Evaluation + reports | Yes | Yes | Yes |
| SFT training | No | No | **Yes** |
| GRPO training | No | No | **Yes** |

### 6.1 Explicit rule
All code outside `training/` must be importable and runnable on both macOS and Windows.

### 6.2 Consequence
Anything CUDA-specific must be isolated behind training-only modules and optional dependencies.

---

## 7. External tools and current grounding

The spec should align to currently available tooling:

- **Unsloth** currently supports running/training Qwen3.5 and exposes Windows install flows in its current README and Studio/Core setup instructions.
- **Unsloth notebooks** include Qwen and GRPO-related notebooks that can be adapted as reference implementations, though some of the listed Qwen3.5 notebooks are vision-oriented and not a direct drop-in for this project.
- **TRL GRPOTrainer** supports multiple reward functions and weighted reward composition.
- **OpenRouter** provides a unified API for many models and supports structured outputs via `response_format`.
- **Ghidra** supports command-line headless analysis through `analyzeHeadless`, which is the practical route for bulk processing.
- **Qwen3.5-4B** is currently available as a Hugging Face model family target and is compatible with standard inference stacks; its native context length is far larger than we need for this prototype.

Because these stacks are moving quickly, the repository must pin versions and include a validated environment manifest for the training path.

---

## 8. Development principles

1. **SFT first, RL second**
   - RL should refine behavior, not teach syntax from scratch.

2. **Binary-grounded prompts**
   - Inputs should contain more than raw decompiler text whenever available.

3. **Structured outputs everywhere**
   - JSON schemas for generated source projects
   - JSON schemas for dataset rows
   - JSON schemas for model outputs in evaluation

4. **Cross-platform core**
   - training is the only CUDA-specific path

5. **Deterministic artifacts**
   - every sample must be reproducible from a manifest

6. **Small, composable CLIs**
   - each phase must be runnable independently

7. **Strict testing**
   - 90% coverage minimum for maintained source code

---

## 9. Canonical repository layout

```text
project-root/
├─ .env.example
├─ .gitignore
├─ .python-version
├─ README.md
├─ spec.md
├─ pyproject.toml
├─ coverage.toml
├─ ruff.toml
├─ mypy.ini
├─ pytest.ini
├─ noxfile.py
├─ configs/
│  ├─ app/
│  │  ├─ default.yaml
│  │  ├─ mac.yaml
│  │  └─ windows.yaml
│  ├─ generation/
│  │  ├─ default.yaml
│  │  ├─ small.yaml
│  │  └─ medium.yaml
│  ├─ compile/
│  │  ├─ clang_o0.yaml
│  │  └─ clang_o2_eval.yaml
│  ├─ ghidra/
│  │  ├─ default.yaml
│  │  └─ export_all.yaml
│  ├─ dataset/
│  │  ├─ sft.yaml
│  │  ├─ rl.yaml
│  │  └─ eval.yaml
│  ├─ training/
│  │  ├─ sft_qwen35_4b.yaml
│  │  ├─ grpo_qwen35_4b.yaml
│  │  ├─ windows_cuda_16gb.yaml
│  │  ├─ windows_cuda_24gb.yaml
│  │  └─ windows_cuda_48gb.yaml
│  └─ prompts/
│     ├─ project_generation.md
│     ├─ function_cleanup.md
│     ├─ rename_only.md
│     └─ readability_judge.md
├─ scripts/
│  ├─ bootstrap.sh
│  ├─ bootstrap.ps1
│  ├─ run_ghidra_export.sh
│  ├─ run_ghidra_export.ps1
│  ├─ train_sft.ps1
│  ├─ train_grpo.ps1
│  └─ smoke_eval.sh
├─ ghidra/
│  ├─ project/
│  ├─ scripts/
│  │  ├─ ExportFunctions.java
│  │  ├─ ExportFunctions.py
│  │  ├─ ExportCallGraph.py
│  │  ├─ ExportStrings.py
│  │  └─ ExportProjectManifest.py
│  └─ templates/
├─ data/
│  ├─ cache/
│  │  ├─ openrouter/
│  │  ├─ compiler/
│  │  └─ ghidra/
│  ├─ raw/
│  │  ├─ generated_projects/
│  │  ├─ binaries/
│  │  ├─ ghidra_exports/
│  │  └─ manifests/
│  ├─ interim/
│  │  ├─ aligned_projects/
│  │  ├─ aligned_functions/
│  │  └─ verified/
│  └─ processed/
│     ├─ sft/
│     ├─ rl/
│     ├─ eval/
│     └─ reports/
├─ artifacts/
│  ├─ models/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ runs/
│  └─ reports/
├─ src/
│  └─ decomp_clarifier/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ settings.py
│     ├─ logging.py
│     ├─ paths.py
│     ├─ schemas/
│     │  ├─ generation.py
│     │  ├─ compiler.py
│     │  ├─ ghidra.py
│     │  ├─ dataset.py
│     │  ├─ model_io.py
│     │  └─ evaluation.py
│     ├─ adapters/
│     │  ├─ openrouter_client.py
│     │  ├─ compiler_clang.py
│     │  ├─ ghidra_headless.py
│     │  ├─ filesystem_cache.py
│     │  └─ subprocess_utils.py
│     ├─ generation/
│     │  ├─ prompt_builder.py
│     │  ├─ project_generator.py
│     │  ├─ validators.py
│     │  └─ canonicalize.py
│     ├─ compilation/
│     │  ├─ build_runner.py
│     │  ├─ test_harness.py
│     │  ├─ compile_db.py
│     │  └─ binary_inventory.py
│     ├─ ghidra_export/
│     │  ├─ export_runner.py
│     │  ├─ parse_exports.py
│     │  └─ aligner.py
│     ├─ dataset/
│     │  ├─ builders.py
│     │  ├─ splitters.py
│     │  ├─ transforms.py
│     │  ├─ prompt_formatter.py
│     │  └─ packers.py
│     ├─ baselines/
│     │  ├─ raw_ghidra.py
│     │  ├─ simple_llm_cleanup.py
│     │  └─ naming_only.py
│     ├─ evaluation/
│     │  ├─ metrics.py
│     │  ├─ compile_eval.py
│     │  ├─ behavior_eval.py
│     │  ├─ readability_eval.py
│     │  ├─ naming_eval.py
│     │  └─ report_builder.py
│     ├─ inference/
│     │  ├─ formatter.py
│     │  ├─ runner.py
│     │  └─ explain.py
│     └─ training/
│        ├─ __init__.py
│        ├─ windows_guard.py
│        ├─ sft/
│        │  ├─ data.py
│        │  ├─ model.py
│        │  ├─ train.py
│        │  └─ callbacks.py
│        ├─ grpo/
│        │  ├─ data.py
│        │  ├─ rewards.py
│        │  ├─ rollout.py
│        │  ├─ verifier.py
│        │  └─ train.py
│        └─ utils/
│           ├─ hardware.py
│           ├─ memory_profiles.py
│           └─ version_lock.py
└─ tests/
   ├─ unit/
   ├─ integration/
   ├─ fixtures/
   ├─ golden/
   └─ smoke/
```

### 9.1 Design notes on the layout
- `src/decomp_clarifier/training/` is the only package allowed to import CUDA-training dependencies.
- Ghidra scripts live outside `src/` because they are executed by Ghidra, not by the main Python runtime.
- Raw artifacts and derived artifacts are separated.
- Every phase writes a manifest.

---

## 10. Python environment strategy

## 10.1 Requirement
Use a **local project venv**, not a global environment.

## 10.2 Canonical path
Use `uv` to create a project-local `.venv`.

### macOS / bash
```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Windows / PowerShell
```powershell
uv venv .venv --python 3.13
.\.venv\Scripts\activate
uv pip install -e ".[dev]"
```

## 10.3 Why `uv`
It matches current Unsloth guidance and keeps installation fast, especially on Windows.

## 10.4 Dependency groups
Use dependency groups in `pyproject.toml`:

- `dev`
- `test`
- `eval`
- `train-windows-cuda`

### Example concept
```toml
[project.optional-dependencies]
dev = ["ruff", "mypy", "pre-commit", "nox"]
test = ["pytest", "pytest-cov", "hypothesis"]
eval = ["pandas", "polars", "matplotlib", "jinja2"]
train-windows-cuda = [
  "unsloth",
  "trl",
  "transformers",
  "datasets",
  "accelerate",
  "peft",
  "bitsandbytes",
]
```

### Hard rule
`pip install -e ".[dev,test,eval]"` must work on both macOS and Windows without requiring CUDA.

---

## 11. Packaging and build rules

1. Use `pyproject.toml` as the single source of packaging truth.
2. Use `src/` layout only.
3. Every CLI entry point must call library code, not shell out from the CLI directly except through adapter layers.
4. Avoid notebook-only logic in core code.
5. Training notebooks are optional references, not the system of record.

---

## 12. Configuration strategy

Use YAML configuration files plus environment variables for secrets.

### 12.1 Secrets
Store in environment variables only:
- `OPENROUTER_API_KEY`

### 12.2 Config precedence
1. command-line flags
2. environment variables
3. chosen YAML config
4. defaults

### 12.3 Mandatory tracked config
Each run writes a frozen config snapshot to:
```text
artifacts/runs/<run_id>/resolved_config.yaml
```

---

## 13. Synthetic data generation via OpenRouter

## 13.1 Why use OpenRouter
OpenRouter is a good fit because it provides a unified API, multiple models behind one endpoint, and structured output support. That lets us:
- swap generation models without rewriting the client
- request strict JSON
- build caching and retries once

## 13.2 Generation objective
Generate **small but semantically meaningful C projects** that:
- compile reliably
- contain multiple functions and helper relationships
- exercise typical decompiler pain points
- remain understandable enough to score later

## 13.3 Content types to generate
The generator should sample project types from a weighted menu:

1. string parsing
2. tokenization
3. simple config parsing
4. checksum / hash-like utilities
5. finite state machines
6. linked-list / array utilities
7. tree traversal
8. binary/hex/text conversion
9. file-format-lite parsing
10. command dispatch tables
11. validation pipelines
12. bit flag manipulation
13. retry / error handling flows
14. struct-heavy data transformations
15. light numerical kernels

## 13.4 What to avoid in v1
- undefined behavior as a feature
- inline assembly
- platform-specific syscalls
- heavy macros
- external dependency trees
- networking and filesystem side effects unless heavily sandboxed
- giant monolithic files
- intentionally malicious payloads

## 13.5 Generation schema
Every OpenRouter generation request should ask for strict JSON with fields like:

```json
{
  "project_id": "string",
  "summary": "string",
  "difficulty": "easy|medium|hard",
  "files": [
    {"path": "src/foo.c", "content": "string"},
    {"path": "src/foo.h", "content": "string"}
  ],
  "tests": [
    {"name": "test_case_name", "input": "string", "expected": "string"}
  ],
  "build": {
    "entrypoints": ["foo.c"],
    "c_standard": "c11",
    "compiler_family": "clang"
  },
  "semantic_hints": {
    "project_purpose": "string",
    "function_intents": [
      {"function_name": "parse_header", "intent": "Parse a compact key/value header"}
    ]
  }
}
```

## 13.6 Prompt strategy
The project generator should request:
- readable original source
- no comments that leak the answer too strongly
- clean identifiers
- deterministic helper structure
- tests
- a high-level semantic description

The semantic description is useful for:
- evaluation
- rename scoring
- optional human-readable reports

## 13.7 Model selection
The OpenRouter client must accept:
- generation model
- fallback models
- temperature
- max tokens
- response format schema

### Default use
Use a strong coding model through OpenRouter for initial synthetic generation. This can be swapped by config without code changes.

## 13.8 Caching
All OpenRouter responses must be cached by:
- request body hash
- chosen model
- schema version

Cache location:
```text
data/cache/openrouter/<hash>.json
```

## 13.9 Validation pipeline
Every generated project must pass:
1. JSON schema validation
2. file path validation
3. compile validation
4. tests validation
5. minimum complexity validation
6. no banned includes or calls
7. reproducibility manifest emission

Projects that fail are discarded or repaired by a separate repair pass.

---

## 14. Compilation pipeline

## 14.1 Compiler choice
Prefer **Clang** in v1.

Reason:
- consistent diagnostics
- good availability on macOS and Windows
- easy control over optimization level and warnings

## 14.2 Compile profiles
Two official profiles:

### Profile A: `clang_o0`
- default profile
- used for most training data
- maximizes alignment simplicity

### Profile B: `clang_o2_eval`
- small challenge eval profile
- not required for all samples
- used to show partial robustness

## 14.3 Compiler output capture
For every project build, capture:
- compiler version
- compile command
- flags
- target triple if available
- produced binary paths
- debug info presence
- strip state

## 14.4 Produced artifacts
Store:
- raw source tree
- build log
- executable or object files
- manifest JSON
- test results

## 14.5 Host-native builds
To keep macOS and Windows support simple in v1:
- compilation is host-native
- dataset rows record `host_os`, `binary_format`, `compiler_version`, and `arch`

We do **not** require identical binaries across OSes.

---

## 15. Ghidra integration

## 15.1 Core requirement
Use **Ghidra headless** for all extraction.

## 15.2 Why headless
Headless analysis is the feasible path for batch processing and reproducible export.

## 15.3 Ghidra script responsibilities
The export script set should extract, per function where possible:

- function address
- function name reported by Ghidra
- signature
- return type
- parameter list
- local variables
- decompiled text
- disassembly text
- strings referenced by the function
- imported functions referenced
- callees
- callers
- basic-block count
- instruction count
- file path / binary path / project id

## 15.4 Mandatory export formats
- JSONL for function rows
- project-level manifest JSON
- optional text snapshots for debugging

## 15.5 Alignment strategy
Align source functions with decompiled functions using:
1. original function names when symbols remain
2. stable function ordering within source file
3. address + file manifest mapping
4. fuzzy backup matching by structure and calls

Because v1 is synthetic and mostly unstripped or lightly stripped, we do not need heroic matching logic.

## 15.6 Strip policy for v1
Use two data modes:
- **symbol-preserving mode** for easier alignment and early SFT
- **partially stripped mode** for harder eval and later RL

This lets us stage the difficulty.

---

## 16. Dataset design

## 16.1 Primary sample granularity
Use **function-level** training samples with **project-level** metadata.

Why:
- many more samples per generated project
- simpler batching
- easier reward evaluation
- still allows project context via neighboring metadata

## 16.2 Secondary sample type
Use **project-level** samples for:
- final evaluation
- demo cases
- limited long-context experiments

## 16.3 Core sample schema
```json
{
  "sample_id": "uuid",
  "project_id": "uuid",
  "split": "train|val|test",
  "task_type": "cleanup|rename|full_clarify",
  "host_os": "windows|macos",
  "compiler": "clang",
  "opt_level": "O0",
  "binary_format": "pe|macho|elf",
  "source_function_name": "parse_header",
  "source_code": "string",
  "target_clean_code": "string",
  "ghidra_function_name": "FUN_140001230",
  "ghidra_decompiled_code": "string",
  "assembly": "string",
  "strings": ["Content-Length", "%d"],
  "imports": ["memcmp", "strlen"],
  "callers": ["..."],
  "callees": ["..."],
  "semantic_summary": "Parse compact header pairs",
  "rename_map_target": {
    "local_18": "header_len",
    "param_1": "input_buf"
  },
  "tests_ref": "path/or/id",
  "difficulty": "medium"
}
```

## 16.4 Task variants
We should materialize three task types from the same underlying data:

### Task A: cleanup
Input:
- Ghidra decompiled code
- assembly
- strings/imports

Output:
- cleaned compilable code

### Task B: rename
Input:
- cleaned or raw decompiled code
- metadata

Output:
- rename map only

### Task C: full clarify
Input:
- raw decompiled code + metadata

Output:
- summary + rename map + cleaned code

## 16.5 Train/val/test split policy
Split by **project**, not function, to avoid leakage.

Recommended starting split:
- 80% train
- 10% validation
- 10% test

## 16.6 Challenge eval splits
Keep separate small eval packs for:
- harder naming
- small `-O2` subset
- partially stripped subset
- long function subset

---

## 17. Target output format

The model should emit a strict JSON object during evaluation.

```json
{
  "summary": "Parses key/value header entries from a compact byte buffer.",
  "confidence": 0.82,
  "renamings": {
    "FUN_140001230": "parse_header_entries",
    "param_1": "input_buf",
    "local_10": "entry_count"
  },
  "cleaned_c": "string"
}
```

### 17.1 Why JSON
- easier reward parsing
- easier validation
- easier regression testing
- better downstream reporting

### 17.2 Training simplification
For SFT, we may store the final assistant output as a single JSON blob or as XML-like tagged sections. The evaluation path should still normalize to the JSON contract above.

---

## 18. Model choice

## 18.1 Primary model
Use **Qwen3.5 4B family** as the default v1 base model.

## 18.2 Rationale
- small enough for local fine-tuning experiments
- supported by current Unsloth materials
- capable enough for code transformation tasks
- large context support exists even if we use much smaller windows for training

## 18.3 Exact checkpoint rule
The training config should not hardcode only one repo string in source code. Instead:
- set checkpoint in config
- allow either:
  - official Qwen checkpoint
  - Unsloth-optimized variant if required by the chosen notebook path

### Canonical default config field
```yaml
model:
  base_model_id: "Qwen/Qwen3.5-4B"
  loader_variant: "unsloth"
```

## 18.4 Practical context length for this project
Even though Qwen3.5 supports much longer native context, v1 training should use:
- 4096 token max sequence for SFT initially
- optional 8192 later if memory allows

Reason:
- decompiler cleanup does not require 128K context
- training stability and throughput matter more than headline context size
- Windows CUDA local training should remain practical

---

## 19. Training strategy overview

## 19.1 Order of operations
1. baseline no-training methods
2. SFT
3. verifier stack
4. GRPO
5. demo and reports

## 19.2 Why this order
SFT should teach:
- output format
- cleanup style
- basic naming behavior
- code reconstruction syntax

GRPO should then optimize:
- digestibility
- consistency
- faithfulness
- verifier-backed correctness

---

## 20. Baselines

We need baselines to prove improvement.

## 20.1 Baseline 0: raw Ghidra
No model. Report raw decompiled output as-is.

## 20.2 Baseline 1: prompt-only cleanup
Use an OpenRouter model to clean raw Ghidra text with no fine-tuning.

## 20.3 Baseline 2: naming-only prompt pass
Use prompt-only rename map generation.

## 20.4 Baseline 3: SFT-only local model
Local fine-tuned model without RL.

## 20.5 Final comparison
Compare:
- raw Ghidra
- prompt-only remote cleanup
- SFT local model
- SFT + GRPO local model

---

## 21. SFT phase

## 21.1 Objective
Teach the model the output contract and the basic cleanup task.

## 21.2 Dataset composition
Recommended starting mixture:
- 50% full clarify
- 30% cleanup-only
- 20% rename-only

## 21.3 Input packing
Prompt should include:
- decompiled function
- optional assembly
- strings/imports
- light call context
- task instruction
- hard output schema

### Example input sections
```text
You are a binary-grounded code clarification assistant.

Task:
Return strict JSON with summary, confidence, renamings, cleaned_c.

Decompiler:
<code>...</code>

Assembly:
<asm>...</asm>

Strings:
["foo", "bar"]

Imports:
["memcmp", "strlen"]

Call context:
{"callers": ["..."], "callees": ["..."]}
```

## 21.4 Target formatting
Train on strict JSON outputs only.

## 21.5 Loss policy
Standard causal LM SFT on assistant output.

## 21.6 Adapter strategy
Use QLoRA/LoRA via Unsloth.

Recommended initial approach:
- 4-bit loading
- LoRA adapters
- start with common linear target modules
- gradient checkpointing enabled if stable

## 21.7 Suggested v1 training profile
This spec intentionally avoids overcommitting to a single exact hyperparameter set, but the initial profile should roughly be:

- model: Qwen3.5 4B
- precision: bf16 if supported, otherwise fp16
- max_seq_length: 4096
- load_in_4bit: true
- LoRA rank: config-driven
- few epochs over the first cleaned dataset
- early stopping on validation readability composite score

## 21.8 Checkpoint policy
Save:
- best by validation composite score
- best by format validity
- last checkpoint

---

## 22. Verifier stack

The verifier stack is critical because RL should not optimize only text similarity.

## 22.1 Verifier responsibilities
1. validate JSON output
2. ensure required keys exist
3. check `cleaned_c` is non-empty
4. reject placeholder-only renames
5. optional syntax parse of `cleaned_c`
6. compile candidate in harness where possible
7. run tests where possible
8. compute naming/readability metrics
9. emit per-sample reward breakdown

## 22.2 Compile harness concept
For function-level evaluation:
- reconstruct a small source harness using:
  - original includes
  - required typedefs/structs
  - function under test
  - minimal wrapper

If compile harness generation is too brittle for a subset, skip compile reward and use readability/naming/format only for those rows.

## 22.3 Behavior verification
For project-level eval or stable subsets:
- compile reconstructed project or reconstructed function harness
- run generated tests or preserved tests
- compare outputs to original behavior

---

## 23. RL phase with GRPO

## 23.1 Objective
Use GRPO to optimize the model beyond imitation:
- better naming
- less confusing structure
- fewer hallucinations
- stronger formatting reliability
- better verifier scores

## 23.2 Why GRPO here
GRPO is a good fit because:
- we can define multiple programmatic reward functions
- we do not need human preference annotation to start
- reward composition is explicit

## 23.3 RL dataset
The RL prompt set should be smaller and more curated than the SFT dataset.

Recommended initial size:
- 500 to 2000 prompts

Each prompt should include:
- decompiled code
- metadata
- task type
- verifier references

## 23.4 Reward composition
Use weighted rewards.

### Proposed initial reward families
1. **format reward**
   - valid JSON
   - required fields present

2. **cleanup reward**
   - fewer placeholder identifiers
   - reduced decompiler noise patterns
   - structured control flow markers improved

3. **naming reward**
   - similarity to target names
   - semantic match heuristics
   - consistency across repeated symbols

4. **compile reward**
   - compiles in harness

5. **behavior reward**
   - test pass / differential equivalence where feasible

6. **readability reward**
   - static heuristics and optional judge score

7. **hallucination penalty**
   - penalize invented imports, calls, globals, or fields unsupported by evidence

## 23.5 Suggested initial weighted reward
```text
R =
  1.0 * format_reward +
  1.5 * cleanup_reward +
  1.5 * naming_reward +
  2.0 * compile_reward +
  3.0 * behavior_reward +
  1.0 * readability_reward -
  2.0 * hallucination_penalty
```

These weights must be config-driven, not hardcoded.

## 23.6 RL rollout constraints
Keep completions short and task-specific:
- JSON only
- moderate max completion tokens
- no chain-of-thought dependence
- reject verbose commentary outside schema

## 23.7 GRPO implementation rule
Implement reward functions as separate pure Python callables whenever possible. The trainer wrapper composes them.

Example module layout:
```text
src/decomp_clarifier/training/grpo/rewards.py
src/decomp_clarifier/training/grpo/verifier.py
src/decomp_clarifier/training/grpo/rollout.py
```

## 23.8 RL safety against reward hacking
Add checks for:
- outputs that trivially copy source target names from leaked context
- empty but JSON-valid responses
- constant summaries
- renaming everything to generic semantic words without justification
- fake compile success flags

---

## 24. Readability and digestibility metrics

Since “digestible” is the actual product goal, we need explicit metrics.

## 24.1 Hard metrics
- JSON validity rate
- field completeness rate
- compile success rate
- test pass rate
- exact name recovery rate
- normalized name similarity
- syntax parse success

## 24.2 Soft metrics
- human readability score
- pairwise preference against raw Ghidra
- pairwise preference against prompt-only baseline
- confidence calibration quality

## 24.3 Heuristic readability indicators
- fewer placeholder names like `var_12`, `local_18`, `iVar3`
- fewer unnecessary temporaries
- shorter and more structured conditionals
- clearer loop boundaries
- consistent naming across parameters and locals
- smaller AST complexity than raw decompiler output when semantics are preserved

## 24.4 Judge-based metric
Optionally use an external judge model to produce pairwise readability comparisons:
- raw Ghidra vs candidate
- prompt baseline vs candidate
- SFT vs SFT+GRPO

The judge must be blinded to which output came from which system.

---

## 25. Data generation phases

## Phase 0: project scaffold
Deliverables:
- repo skeleton
- local `.venv` scripts
- CLI shell
- config loading
- schema validation
- test harness
- 90% coverage gate wired

## Phase 1: OpenRouter generation
Deliverables:
- prompt templates
- strict JSON schema generation
- cache
- validation
- 100 to 300 valid projects

## Phase 2: compile + Ghidra export
Deliverables:
- compile manifests
- Ghidra scripts
- headless orchestration
- aligned project manifests

## Phase 3: dataset builder
Deliverables:
- function-level rows
- train/val/test split
- SFT JSONL / parquet
- RL prompt set
- baseline eval pack

## Phase 4: baseline evaluation
Deliverables:
- raw Ghidra baseline report
- prompt-only report
- first metrics dashboard

## Phase 5: SFT
Deliverables:
- local Windows CUDA training path
- saved adapters/checkpoints
- eval report against baselines

## Phase 6: verifier + GRPO
Deliverables:
- reward breakdown
- RL prompt set
- GRPO run
- SFT vs RL comparison

## Phase 7: demo + final report
Deliverables:
- CLI inference command
- report HTML/Markdown
- curated examples
- before/after gallery

---

## 26. CLI design

Use a single Typer-based CLI.

### Proposed commands
```text
decomp-clarifier generate-projects
decomp-clarifier compile-projects
decomp-clarifier export-ghidra
decomp-clarifier build-dataset
decomp-clarifier run-baselines
decomp-clarifier train-sft
decomp-clarifier train-grpo
decomp-clarifier eval
decomp-clarifier report
decomp-clarifier demo
```

### Important rule
Training commands should fail fast on non-Windows or non-CUDA systems with a clear message.

---

## 27. Windows CUDA training path

## 27.1 Primary target
Native Windows + NVIDIA CUDA + local `.venv`.

## 27.2 Why native Windows
The user explicitly wants the training path tuned for CUDA on Windows.

## 27.3 Practical caution
Because Unsloth, TRL, Triton, PyTorch, and Qwen3.5 support continue to evolve quickly, we must:
- pin a validated version set
- keep a `version_lock.py` check
- maintain a `windows_cuda_validated.md` note
- support a quick environment smoke test before full training

## 27.4 Required environment checks
Before training:
- GPU visible via torch
- CUDA available
- bf16 support flag recorded
- package versions logged
- tiny SFT smoke batch runs
- tiny GRPO smoke batch runs

## 27.5 Training environment manifest
Write:
```text
artifacts/runs/<run_id>/environment.json
```

Include:
- OS version
- Python version
- torch version
- CUDA version
- unsloth version
- trl version
- transformers version
- GPU name and VRAM

## 27.6 Recommended operational stance
Maintain one known-good pinned environment. Do not chase latest package versions by default.

---

## 28. Non-training portability rules

The following modules must have **no training dependency imports**:
- `generation`
- `compilation`
- `ghidra_export`
- `dataset`
- `baselines`
- `evaluation`
- `inference`

If a training-related object is needed for type hints, use `TYPE_CHECKING` guards or protocol types.

---

## 29. Testing strategy

## 29.1 Coverage target
At least **90% test coverage** for maintained Python code under `src/`.

### Exclusions allowed
Exclude from coverage:
- Ghidra script files executed externally
- generated files under `data/`
- notebooks
- one-off shell wrappers
- experimental prototypes not imported by production code

## 29.2 Test layers
### Unit tests
Cover:
- schemas
- config loading
- prompt builders
- cache keys
- parsers
- reward functions
- metrics

### Integration tests
Cover:
- OpenRouter response parsing with fixtures
- compilation pipeline on tiny projects
- Ghidra export parser with golden files
- dataset assembly from fixtures
- report generation

### Smoke tests
Cover:
- end-to-end tiny pipeline with 1 or 2 toy projects
- no real training required
- optional Windows CUDA smoke for training path

## 29.3 Test policy
Every bug fix must include:
- one regression test
- one note in changelog or issue reference

## 29.4 Coverage enforcement
CI should fail below 90%.

Example:
```bash
pytest --cov=src/decomp_clarifier --cov-report=term-missing --cov-fail-under=90
```

## 29.5 Golden tests
Store stable fixtures for:
- Ghidra export rows
- assembled dataset rows
- evaluation reports
- model output normalization

---

## 30. Quality gates

A phase is only “done” if it passes its gates.

## 30.1 Phase 0 gate
- install works in local `.venv` on macOS and Windows
- lint, typecheck, tests all green
- coverage >= 90%

## 30.2 Phase 1 gate
- at least 100 valid compilable generated projects
- cache and manifest work
- invalid generations are quarantined

## 30.3 Phase 2 gate
- at least 95% of valid compiled projects produce parseable Ghidra exports
- per-project manifest includes binary and export references

## 30.4 Phase 3 gate
- project-level split leakage test passes
- dataset rows schema-valid
- task distributions are reported

## 30.5 Phase 4 gate
- baseline report generated
- at least 20 curated before/after examples stored

## 30.6 Phase 5 gate
- SFT run completes on validated Windows CUDA environment
- SFT beats raw Ghidra on readability
- SFT maintains near-perfect format validity

## 30.7 Phase 6 gate
- GRPO run completes
- GRPO beats SFT on at least one primary metric without collapsing compile/test metrics

---

## 31. CI and automation

## 31.1 CI matrix
CI should run on:
- macOS latest
- Windows latest

But **training is not run in CI** unless GPU infrastructure is explicitly added later.

## 31.2 CI jobs
1. lint
2. typecheck
3. unit tests
4. integration tests
5. coverage gate
6. package import smoke
7. tiny dataset build smoke

## 31.3 Optional local-only jobs
- `nox -s train_smoke_windows_cuda`
- `nox -s ghidra_smoke`

---

## 32. Logging, tracking, and reproducibility

## 32.1 Mandatory per-run outputs
For every major run:
- run id
- git commit hash if available
- resolved config
- environment manifest
- input manifest
- output summary metrics

## 32.2 Log structure
```text
artifacts/logs/<run_id>.log
artifacts/runs/<run_id>/metrics.json
artifacts/runs/<run_id>/resolved_config.yaml
artifacts/runs/<run_id>/environment.json
```

## 32.3 Randomness control
Record:
- Python seed
- model generation seed when supported
- dataset split seed

---

## 33. Reporting

## 33.1 Report artifacts
Generate:
- Markdown summary
- HTML report
- CSV or parquet metrics export

## 33.2 Must-have report sections
1. dataset summary
2. baseline comparison
3. SFT comparison
4. GRPO comparison
5. representative wins
6. failure cases
7. reward breakdown
8. environment and version table

## 33.3 Curated examples
Each report should include before/after examples:
- raw decompiled code
- model output
- original source
- short note on what improved

---

## 34. Risks and mitigations

## 34.1 Risk: synthetic data is too neat
Mitigation:
- vary prompt styles
- request different coding idioms
- add small noise variants
- include a harder eval split

## 34.2 Risk: model overfits source formatting
Mitigation:
- normalize source formatting
- use multiple task types
- compare against readability and compile metrics, not just text similarity

## 34.3 Risk: RL reward hacking
Mitigation:
- use verifier checks
- separate reward components
- inspect failure examples
- include hallucination penalties

## 34.4 Risk: Windows CUDA environment drift
Mitigation:
- pin validated versions
- smoke test before full runs
- keep training dependency boundary strict

## 34.5 Risk: compile harness brittleness
Mitigation:
- use compile reward only on stable subsets at first
- keep format/naming/readability rewards active for the rest

---

## 35. Initial implementation recommendations

## 35.1 Start tiny
First real target:
- 100 to 200 generated projects
- mostly `-O0`
- function-level dataset
- SFT only
- no RL until baseline + SFT comparison works

## 35.2 First milestone definition
A milestone is successful if:
- the system ingests raw Ghidra exports
- the SFT model outputs valid JSON consistently
- humans prefer SFT output over raw Ghidra in a blinded sample

## 35.3 RL entry condition
Do not start GRPO until:
- JSON validity > 98%
- dataset pipeline is stable
- baseline report exists
- SFT model already produces plausible cleaned code

---

## 36. What “done” means for v1

The project is done when all of the following are true:

1. A user can generate synthetic projects through OpenRouter.
2. The system compiles them and decompiles them via Ghidra headless.
3. The dataset builder creates aligned function-level training rows.
4. A Windows CUDA training run fine-tunes Qwen3.5 with Unsloth.
5. A second-stage GRPO run completes on a curated RL dataset.
6. Evaluation reports show the tuned model produces outputs that are more digestible than raw Ghidra and better than prompt-only cleanup on a meaningful subset.
7. The repo has at least 90% test coverage for maintained code.
8. All non-training pipeline pieces run on macOS and Windows.

---

## 37. Suggested implementation sequence by week or sprint

### Sprint 1
- scaffold repo
- env bootstrap
- schemas
- CLI skeleton
- tests and coverage gate

### Sprint 2
- OpenRouter adapter
- generation prompt
- cache
- compile validator
- first 50 valid projects

### Sprint 3
- Ghidra headless runner
- export parsers
- aligned manifests
- tiny dataset builder

### Sprint 4
- baseline evaluators
- report builder
- 100 to 200 project corpus

### Sprint 5
- SFT data packer
- Unsloth SFT training on Windows CUDA
- first local checkpoint

### Sprint 6
- verifier stack
- GRPO reward functions
- curated RL split

### Sprint 7
- GRPO run
- side-by-side report
- curated examples and failure analysis

---

## 38. Concrete engineering conventions

1. Type hints required in maintained Python modules.
2. Pydantic models for all external data contracts.
3. No untyped dict plumbing across module boundaries.
4. One public class or concept per file when possible.
5. No hardcoded paths outside `paths.py`.
6. Every CLI subcommand must emit a manifest or metrics file.
7. No direct `print` in library code; use structured logging.
8. All subprocess calls go through a single utility wrapper.
9. All external adapters must have fixture-based tests.

---

## 39. Notes on notebook use

Use Unsloth notebooks as **reference material**, not as the main implementation surface.

Practical rule:
- any logic copied from notebook experimentation must be moved into `src/decomp_clarifier/training/...`
- notebooks may exist under `notebooks/` later, but production runs use config-driven Python entry points

---

## 40. References to ground the implementation

These are the key current references that justify the spec direction:

1. Unsloth current repo / README  
   https://github.com/unslothai/unsloth

2. Unsloth notebook collection  
   https://github.com/unslothai/notebooks

3. TRL GRPOTrainer docs  
   https://huggingface.co/docs/trl/grpo_trainer

4. TRL rewards docs  
   https://huggingface.co/docs/trl/rewards

5. OpenRouter quickstart  
   https://openrouter.ai/docs/quickstart

6. OpenRouter Python SDK docs  
   https://openrouter.ai/docs/sdks/python/overview

7. OpenRouter API reference and structured outputs  
   https://openrouter.ai/docs/api/reference/overview

8. Ghidra command-line / headless example  
   https://ghidra.re/ghidra_docs/GhidraClass/BSim/BSimTutorial_Ghidra_Command_Line.html

9. Ghidra HeadlessAnalyzer API  
   https://ghidra.re/ghidra_docs/api/ghidra/app/util/headless/HeadlessAnalyzer.html

10. Qwen3.5-4B model card  
    https://huggingface.co/Qwen/Qwen3.5-4B
