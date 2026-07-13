# Test Instructions

Scope: `tests/`

- Prefer deterministic and fast unit tests.
- Use synthetic fixtures only.
- Never use real books, private documents, prompts, indexes, or logs.
- Do not download models during normal test execution.
- Mock filesystem, model, network, and provider boundaries where appropriate.
- Separate hardware-dependent and model-dependent integration tests from unit tests.
- Mark integration tests clearly.
- Tests must not modify directories outside temporary test locations.
- Verify exit codes and user-visible CLI output where relevant.
