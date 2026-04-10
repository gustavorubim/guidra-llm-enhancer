# AGENTS.md

## 1. Purpose

This file defines how coding agents should work in this repository.

The repository implements a **binary-grounded decompiler clarification system**. The goal is not to reconstruct original source perfectly. The goal is to show that confusing Ghidra decompiler output can be transformed into code that is materially easier for a human to read, review, and reason about.

Agents must optimize for:
- readability
- semantic plausibility
- clearer naming
- better structure
- reproducibility
- small, testable increments

Agents must not optimize for:
- exact source recovery
- broad compiler coverage in v1
- broad architecture coverage in v1
- large real-world codebase support in v1
- unsupported CUDA workflows outside the Windows training path

## 2. Source of truth

When conflicts exist, resolve them in this order:

1. `spec.md`
2. `AGENTS.md`
3. tracked config files in `configs/`
4. tests and golden fixtures
5. inline code comments

If a requested implementation conflicts with `spec.md`, follow `spec.md` unless the user explicitly changes the spec.

## 3. Product framing

This project is a **post-processor for decompiler output**.

Canonical input:
- Ghidra decompiled C
- assembly
- strings and imports
- light call-context metadata

Canonical output:
- cleaner C-like code
- improved function and variable names
- simplified control flow where justified
- short natural-language summary
- optional rename map
- optional confidence notes

Agents should preserve this framing in code, prompts, tests, docs, and demos.

## 4. Platform rules

### 4.1 Supported platforms
- macOS: all non-training workflows
- Windows: all non-training workflows
- Windows + NVIDIA CUDA: training workflows

### 4.2 Hard boundary
Everything outside `src/decomp_clarifier/training/` must work on both macOS and Windows.

### 4.3 CUDA rule
Only training modules may import or require CUDA-specific dependencies.

### 4.4 Local environment rule
Use a project-local `.venv` only. Do not assume global Python packages.

Preferred bootstrap:

#### macOS
```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,test,eval]"
```

#### Windows PowerShell
```powershell
uv venv .venv --python 3.13
.\.venv\Scripts\activate
uv pip install -e ".[dev,test,eval]"
```

For training on Windows CUDA:

```powershell
uv pip install -e ".[dev,test,eval,train-windows-cuda]"
```

## 5. v1 scope guardrails

Agents must keep v1 intentionally narrow.

### 5.1 Canonical v1 assumptions
- language: C
- compiler family: Clang preferred
- optimization: mostly `-O0`
- optional eval challenge: small curated `-O2` split
- project size: 1 to 8 source files
- function count: 3 to 20 per project
- source origin: synthetic projects generated through OpenRouter
- target model family: Qwen3.5 4B family
- fine-tuning framework: Unsloth
- RL phase: GRPO after SFT only

### 5.2 Out of scope unless the user expands scope
- generalized multi-compiler support
- generalized multi-architecture support
- malware-specific workflows
- perfect identifier recovery
- arbitrary project-scale recompilation of recovered output

## 6. Repository map and ownership

Canonical layout:

```text
configs/          Config files
scripts/          Human-friendly bootstrap and run scripts
ghidra/           Ghidra scripts and templates
data/             Raw, interim, processed datasets and caches
artifacts/        Models, checkpoints, reports, run outputs
src/              Application code
tests/            Unit, integration, smoke, fixtures, golden tests
```

### 6.1 Ownership by area
- `src/decomp_clarifier/adapters/`: external tool and API boundaries
- `src/decomp_clarifier/generation/`: synthetic project generation and validation
- `src/decomp_clarifier/compilation/`: compile and test harness
- `src/decomp_clarifier/ghidra_export/`: export parsing and alignment
- `src/decomp_clarifier/dataset/`: dataset assembly and prompt formatting
- `src/decomp_clarifier/baselines/`: non-fine-tuned baselines
- `src/decomp_clarifier/evaluation/`: metrics, judge calls, reports
- `src/decomp_clarifier/inference/`: local model and prompt inference flows
- `src/decomp_clarifier/training/`: Windows CUDA training only

Agents must preserve this separation.

## 7. Agent roster

The project may be implemented by one agent or several. These roles define responsibilities and handoff contracts.

### 7.1 Orchestrator Agent
Owns:
- task decomposition
- sequencing by phase
- cross-module coordination
- run manifest integrity
- acceptance checks before handoff

Must ensure:
- each phase is independently runnable
- config snapshots are written per run
- artifacts are traceable back to inputs and configs
- no module crosses platform boundaries incorrectly

Deliverables:
- phase plans
- implementation tickets
- resolved config snapshots
- run summaries

### 7.2 Generation Agent
Owns:
- OpenRouter client integration
- prompt construction for project generation
- structured response parsing
- generated project validation and canonicalization
- cache keys and cache reuse

Inputs:
- generation config
- prompt template
- OpenRouter API key

Outputs:
- generated C project files
- generation manifest
- validation report
- cached request and response artifacts

Rules:
- require structured JSON output when possible
- reject malformed or non-compiling project payloads
- keep v1 generation small and semantically meaningful
- do not widen scope to large realistic repos unless requested

### 7.3 Compilation Agent
Owns:
- host-native compilation
- compile profile execution
- unit test execution for generated projects
- compiler output capture
- binary inventory manifests

Inputs:
- generated source project
- compile config

Outputs:
- build logs
- test logs
- binaries
- compile database or equivalent metadata
- manifest of source to binary mappings

Rules:
- prefer Clang for v1
- `clang_o0` is the default path
- `clang_o2_eval` is optional and must stay isolated as a challenge split
- failures must be preserved as artifacts, not swallowed

### 7.4 Ghidra Export Agent
Owns:
- headless Ghidra invocation
- project import and analysis
- export of decompiled functions, assembly, strings, call graph, and manifests
- export parsing
- source-to-export alignment helpers

Inputs:
- compiled binary
- ghidra config
- ghidra scripts

Outputs:
- raw Ghidra export files
- normalized parsed export records
- alignment diagnostics

Rules:
- use headless analysis only for automated flows
- keep Ghidra-executed scripts outside the main Python package
- capture enough metadata to support later debugging of bad alignments

### 7.5 Dataset Agent
Owns:
- aligned project and function sample creation
- task-specific transformations
- train, val, test splitting
- challenge split creation
- prompt packing and target formatting

Inputs:
- validated source projects
- compilation outputs
- Ghidra exports

Outputs:
- SFT dataset rows
- RL prompt dataset rows
- eval dataset rows
- split manifests
- task-specific prompt packs

Rules:
- primary sample unit is the function
- keep project-level context links available
- preserve deterministic IDs
- all output rows must be schema-validated

### 7.6 Baseline Agent
Owns:
- raw Ghidra baseline
- prompt-only cleanup baseline
- naming-only baseline
- SFT-only baseline comparisons

Inputs:
- eval dataset
- baseline config

Outputs:
- baseline predictions
- baseline metrics
- example reports

Rules:
- baselines must be reproducible and cheap enough to rerun
- never compare fine-tuned results only against hand-picked examples
- keep baseline interfaces compatible with final evaluation reports

### 7.7 Training Agent
Owns:
- Windows CUDA environment checks
- Unsloth model setup
- Qwen3.5 checkpoint selection
- SFT training
- GRPO training after SFT
- checkpoint packaging

Inputs:
- training config
- processed datasets
- validated Windows CUDA environment

Outputs:
- adapter checkpoints
- training logs
- memory profiles
- training manifests

Rules:
- training happens only on Windows with NVIDIA CUDA
- SFT must precede RL
- training code must be isolated under `src/decomp_clarifier/training/`
- version pinning is required for all training dependencies
- notebook code is never the source of truth

### 7.8 Verifier and Reward Agent
Owns:
- compile checks for model outputs
- behavior checks and differential checks where applicable
- structural and readability heuristics
- naming and consistency scoring
- GRPO reward composition

Inputs:
- candidate clarified outputs
- verifier config
- eval harnesses

Outputs:
- verifier pass and fail records
- reward vectors
- per-sample diagnostics

Rules:
- behavior and compile validity outrank style
- rewards must be difficult to game
- reward weights must be config-driven
- verifier failures must be inspectable

### 7.9 Evaluation Agent
Owns:
- metric computation
- judge-based readability scoring
- comparison across baselines, SFT, and GRPO
- artifact-rich final reports

Inputs:
- predictions from all systems
- eval dataset
- metric config

Outputs:
- machine-readable metrics
- HTML or Markdown reports
- curated examples
- final summary tables

Rules:
- use both hard metrics and soft digestibility metrics
- keep curated examples representative, not cherry-picked
- preserve raw predictions for auditability

### 7.10 QA and Release Agent
Owns:
- linting
- typing
- test coverage
- smoke tests
- release readiness
- CI expectations

Inputs:
- current repo state

Outputs:
- test reports
- coverage reports
- release checklist status

Rules:
- minimum coverage is 90%
- maintain fast unit tests and targeted integration tests
- training is excluded from default macOS and Windows CPU test environments unless explicitly requested

## 8. Phase plan and expected outputs

Agents should work in this order.

### Phase 0: scaffold
Expected outputs:
- repo scaffold
- packaging and config skeleton
- bootstrap scripts
- basic CI and test harness

Gate:
- editable install works on macOS and Windows
- lint, typing, and tests run locally

### Phase 1: OpenRouter generation
Expected outputs:
- OpenRouter adapter
- prompt templates
- structured generation schema
- validated small C project generation

Gate:
- sample projects compile reliably
- generation is cached and reproducible

### Phase 2: compile and Ghidra export
Expected outputs:
- compile runner
- test runner
- headless Ghidra runner
- export parsers

Gate:
- source, binaries, and Ghidra artifacts are linked by manifest

### Phase 3: dataset builder
Expected outputs:
- aligned function dataset
- SFT, RL, and eval dataset builders
- deterministic splits

Gate:
- all rows validate against schema
- splits are reproducible

### Phase 4: baselines
Expected outputs:
- raw Ghidra baseline
- prompt-only cleanup baseline
- naming-only baseline
- report builder for comparisons

Gate:
- baseline metrics and examples can be generated end to end

### Phase 5: SFT
Expected outputs:
- Windows-only training flow
- Qwen3.5 + Unsloth SFT run
- checkpoint packaging

Gate:
- SFT outperforms raw Ghidra and prompt-only baseline on digestibility-oriented metrics

### Phase 6: verifier and GRPO
Expected outputs:
- verifier stack
- GRPO reward config
- RL fine-tuning flow

Gate:
- GRPO improves at least one target metric without unacceptable regressions in compile or behavior checks

### Phase 7: demo and report
Expected outputs:
- demo CLI
- final evaluation report
- curated examples

Gate:
- end-to-end demonstration is reproducible from tracked configs and manifests

## 9. Engineering rules

### 9.1 General coding rules
- use Python with `src/` layout
- prefer small, typed modules
- separate pure logic from side effects
- prefer dataclasses or Pydantic-style schema models for external boundaries
- keep CLIs thin
- centralize subprocess behavior in adapter utilities
- avoid hidden global state

### 9.2 Dependency rules
- core code must not depend on training-only packages
- put optional dependencies behind extras
- version-pin training dependencies
- do not add notebook-only or GUI-only dependencies to core paths

### 9.3 Manifest rules
Every run must write enough data to reproduce itself. At minimum:
- resolved config
- inputs and output paths
- package versions where relevant
- random seeds where relevant
- hashes or IDs for generated samples

### 9.4 Logging rules
- structured logs preferred
- log external tool commands at debug level
- preserve stdout and stderr for compiler, Ghidra, and training runs
- never discard failure artifacts

## 10. Testing policy

### 10.1 Coverage target
Maintain at least **90% test coverage** for maintained source code.

### 10.2 Allowed exclusions
Coverage exclusions may be limited to:
- thin CLI wrappers
- `if __name__ == "__main__"` guards
- platform guards that cannot execute on the current host
- minimal training bootstrap code where direct coverage is impractical

Do not exclude business logic to game coverage.

### 10.3 Test layers
#### Unit tests
Cover:
- schema validation
- prompt builders
- canonicalizers
- cache keys
- manifest writing
- parsing utilities
- metric logic

#### Integration tests
Cover:
- OpenRouter client mocking and schema handling
- compile pipeline on a toy fixture
- Ghidra export parser on stored fixture outputs
- dataset builder from fixture artifacts
- report generation from fixture predictions

#### Smoke tests
Cover:
- end-to-end non-training path on tiny fixtures
- CLI command sanity
- environment bootstrap assumptions

### 10.4 Golden tests
Use golden fixtures for:
- prompt formatting
- parsed Ghidra exports
- evaluation reports
- model output post-processing

## 11. CLI expectations

The CLI should expose small, composable commands. Target command families:

```text
decomp-clarifier generate
decomp-clarifier compile
decomp-clarifier ghidra-export
decomp-clarifier build-dataset
decomp-clarifier run-baseline
decomp-clarifier evaluate
decomp-clarifier infer
decomp-clarifier train-sft
decomp-clarifier train-grpo
```

Rules:
- each command must be independently runnable
- commands must call library code, not embed all logic inline
- training commands must fail fast on unsupported platforms

## 12. Prompt and output rules

### 12.1 Prompt principles
- binary-grounded inputs beat decompiler-only inputs
- structured outputs are preferred
- prompts must be versioned and tracked in `configs/prompts/`
- prompt-only baselines must use the same general framing as the trained system

### 12.2 Output principles
Prefer structured model outputs that can be validated. Canonical fields:
- `summary`
- `rename_map`
- `clarified_code`
- `confidence_notes`

During training, a simplified target may be used if it improves stability, but evaluation should still support the richer schema.

## 13. Data handling rules

### 13.1 Synthetic source generation
Use OpenRouter-backed LLM generation for v1. Keep generation diverse enough to cover small but meaningful C patterns, but do not chase massive realism in v1.

### 13.2 Cached artifacts
Cache the following:
- OpenRouter requests and responses
- compile artifacts
- Ghidra export artifacts
- processed dataset rows where regeneration is expensive

### 13.3 Deterministic IDs
Every project and function sample must have stable IDs derived from manifests or content hashes.

## 14. Training-specific rules

### 14.1 Model choice
Default target: Qwen3.5 4B family as defined in `spec.md` and tracked training configs.

### 14.2 Training order
1. build datasets
2. run baselines
3. run SFT
4. validate SFT improvement
5. build verifier stack
6. run GRPO

### 14.3 RL policy
Do not start GRPO until:
- SFT works end to end
- verifier outputs are stable
- reward functions are inspectable and logged

### 14.4 Reward priority
When reward components conflict, prioritize:
1. compile validity
2. behavior or semantic faithfulness
3. structural clarity
4. naming quality
5. style polish

## 15. Security and scope hygiene

Agents must keep this repository in the lane defined by the spec.

Allowed:
- synthetic code generation
- compile and decompile workflows
- clarity-focused transformation and evaluation
- verifier-backed readability research

Not default v1 work:
- offensive exploit development
- malware automation
- arbitrary real-binary triage pipelines
- unsupported external execution environments

## 16. Definition of done for implementations

A change is done only when all of the following hold:
- code is typed and lint-clean
- tests pass
- coverage stays at or above 90%
- config and artifact paths are documented
- platform boundaries are respected
- run manifests are written
- docs are updated if behavior changed

## 17. Pull request or handoff checklist

Before handoff, agents must verify:
- [ ] implementation follows `spec.md`
- [ ] platform assumptions are explicit
- [ ] non-training code runs on macOS and Windows
- [ ] training code is isolated to `training/`
- [ ] new configs have sensible defaults
- [ ] tests cover success and failure paths
- [ ] coverage remains at or above 90%
- [ ] logs and manifests are written for new run types
- [ ] docs or README snippets are updated where needed

## 18. Preferred implementation sequence

When starting from an empty repo, agents should build in this order:

1. scaffold, packaging, config loading, logging, paths
2. schemas and manifest models
3. OpenRouter adapter and generation validation
4. compile runner and toy fixture tests
5. Ghidra runner and parser fixtures
6. dataset builder and split logic
7. baseline evaluation and report generation
8. Windows guard and SFT pipeline
9. verifier stack
10. GRPO pipeline
11. demo CLI and final report polish

## 19. Final instruction to agents

Keep the project small, grounded, and demonstrable.

This repository is successful if it shows, with reproducible evidence, that confusing Ghidra output can be transformed into something more digestible for a human reviewer. It does not need to solve general binary-to-source reconstruction.
