# Project workflow instructions

When assisting with this repository:

- Treat `ice-shelf-different-runs` as the active development branch unless the user says
  otherwise.
- Before launching or recommending a model run, add or update its entry in `RUNS.md`.
- For every run command or submission instruction, explicitly check the CFL controls and
  expected operating CFL. Keep the sustained production-run CFL at or above 0.1 and no
  higher than 0.5, normally targeting 0.3--0.4. A temporarily lower CFL during weak-flow
  initial spin-up is acceptable, but do not leave a production run below 0.1 without
  explaining why. Adjust `max_dt` gradually and preserve the upper stability limit.
- Record the exact Git commit, run command/settings, machine or job ID, output location,
  status, and scientific outcome when known.
- Never invent missing run metadata. Mark name-based conclusions as inferred and ask the
  user to confirm them when they affect interpretation.
- Keep source code, small configuration files, and the run registry in GitHub.
- Keep NetCDF files, checkpoints, movies, and other large generated output out of GitHub.
- Before committing, review changes for generated output, credentials, and accidental files.
- Use descriptive commit messages and push only after the user requests a GitHub sync.
- Do not merge into `main`, rewrite history, or delete run output unless the user explicitly
  requests it.
