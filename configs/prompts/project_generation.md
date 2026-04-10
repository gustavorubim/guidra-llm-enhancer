You are generating a compact, readable C project for a binary-grounded decompiler clarification dataset.

Return strict JSON that matches the repository schema.

Constraints:
- generate semantically meaningful C11 code
- keep the project small and compilable
- include tests with deterministic input/output pairs
- make the compiled entrypoint consume stdin and print exactly the output expected by the listed tests
- ensure the listed `tests` match the behavior of the compiled program, not just helper functions
- avoid platform-specific syscalls and external dependencies
- include multiple helper functions with clear relationships
- avoid comments that reveal the answer too strongly

Topic menu:
{topics}

Requested difficulty distribution:
{difficulty_weights}

Validation rules:
{validation_rules}
