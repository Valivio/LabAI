# Source Code Instructions

Scope: `src/`

- Implement only approved functionality.
- Inspect existing modules and public interfaces before editing.
- Preserve CLI behavior unless explicitly instructed otherwise.
- Keep configuration access centralized.
- Keep MLX and other inference backends behind internal interfaces.
- Keep external cloud providers behind replaceable adapters.
- Avoid speculative base classes, managers, factories, and plugin systems.
- Prefer small modules with explicit responsibilities.
- Use type hints for public functions and important internal boundaries.
- Add tests for every changed behavior.
- Do not access private data or model directories in unit tests.
