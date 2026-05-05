do not use emdashes or ---

# Pixi CLI Quick Reference

Pixi is a fast, cross-platform package manager built on the conda ecosystem. It manages project dependencies via a `pixi.toml` (or `pyproject.toml`) manifest and a `pixi.lock` lockfile.

## Install

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | sh

# Windows
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"

# macOS (Homebrew)
brew install pixi

# Update
pixi self-update
```

Binary installs to `~/.pixi/bin`. Restart your shell after install.

## Core Concepts

- **Workspace**: A directory with a `pixi.toml` (or `pyproject.toml`) manifest.
- **Environment**: A resolved set of packages from the manifest. A workspace can have multiple environments.
- **Lock file**: `pixi.lock` pins exact versions for reproducibility. Commit this to version control.
- **Tasks**: Named commands defined in the manifest, run inside the environment.
- **Channels**: Conda package sources (default: `conda-forge`). PyPI packages also supported.

## Project Lifecycle

### Create a workspace

```bash
pixi init                        # current dir, creates pixi.toml
pixi init myproject              # in ./myproject
pixi init --format pyproject     # use pyproject.toml instead
pixi init -c conda-forge -c bioconda  # specify channels
pixi init -p linux-64 -p osx-arm64    # specify platforms
pixi init -i environment.yml    # import from conda env file
```

### Add & remove dependencies

```bash
pixi add numpy pandas            # conda packages
pixi add "pytorch>=2.0"          # with version constraint
pixi add --pypi requests         # PyPI package
pixi add --pypi "flask>=3.0"     # PyPI with constraint
pixi add --build cmake           # build dependency
pixi add --host python           # host dependency
pixi add -e test pytest          # add to specific environment
pixi add -p linux-64 cuda        # platform-specific dep

pixi remove numpy                # remove a dependency
pixi remove --pypi requests      # remove PyPI dep
```

### Install & lock

```bash
pixi install                     # solve + install environment
pixi lock                        # solve + update lockfile only (no install)
pixi reinstall                   # force re-solve and re-install
```

### Run commands

```bash
pixi run python script.py        # run arbitrary command in env
pixi run start                   # run a named task
pixi run -e test pytest          # run in specific environment
```

`pixi run` auto-installs the environment if needed before executing.

### Enter a shell

```bash
pixi shell                       # interactive shell with env activated
pixi shell -e cuda               # shell in specific environment
# type `exit` to leave
```

### One-off execution (no project needed)

```bash
pixi exec python                 # run python from a temp env
pixi exec --spec "python>=3.12" python  # with version spec
```

## Task Management

Define tasks in `pixi.toml`:

```toml
[tasks]
start = "python main.py"
lint = "ruff check ."
test = { cmd = "pytest", depends-on = ["lint"] }
```

Manage via CLI:

```bash
pixi task add start "python main.py"
pixi task add test "pytest" --depends-on lint
pixi task add greet "echo hello" --env dev
pixi task remove start
pixi task alias all test lint     # alias that runs multiple tasks
pixi task list                    # show all tasks
```

Run tasks:

```bash
pixi run start
pixi run test                    # runs lint first (dependency)
```

## Inspect Workspace

```bash
pixi list                        # packages in current env
pixi list -e cuda                # packages in specific env
pixi tree                        # dependency tree
pixi info                        # workspace & system info
pixi search numpy                # search conda for a package
```

## Update Dependencies

```bash
pixi update                      # update lockfile to latest allowed versions
pixi update numpy                # update specific package
pixi upgrade                     # upgrade manifest AND lockfile to latest
pixi upgrade numpy               # upgrade specific package in manifest
```

`update` respects existing version constraints. `upgrade` rewrites constraints to allow newer versions.

## Global Tools

Install CLI tools globally (not tied to a project):

```bash
pixi global install ruff         # install and expose `ruff` binary
pixi global install "python>=3.12"
pixi global list                 # list global envs and binaries
pixi global uninstall ruff       # remove
pixi global sync                 # sync manifest with installed envs
```

Global installs live in `~/.pixi` (override with `PIXI_HOME`).

## Multi-Environment

Define environments in `pixi.toml`:

```toml
[feature.test.dependencies]
pytest = "*"

[feature.cuda.dependencies]
cuda = ">=12"

[environments]
test = ["test"]
cuda = ["cuda"]
```

Target specific environments:

```bash
pixi run -e test pytest
pixi shell -e cuda
pixi add -e test hypothesis
pixi list -e cuda
```

## Multi-Platform

```toml
[project]
platforms = ["linux-64", "osx-arm64", "win-64"]

[target.linux-64.dependencies]
cuda = ">=12"

[target.osx-arm64.dependencies]
metal-cpp = "*"
```

```bash
pixi add -p linux-64 cuda        # add platform-specific dep
```

## Useful Options (all commands)

| Flag | Effect |
|------|--------|
| `-m, --manifest-path <PATH>` | Point to a specific manifest file |
| `-e, --environment <ENV>` | Target a named environment |
| `-p, --platform <PLATFORM>` | Target a specific platform |
| `-v` / `-vv` / `-vvv` | Increase verbosity |
| `-q` | Quiet mode |
| `--no-progress` | Suppress progress bars |
| `--frozen` | Don't update lockfile, fail if out of date |
| `--locked` | Don't update lockfile, error if it would change |

## Cleanup

```bash
pixi clean                       # remove installed environments
pixi clean cache                 # clear package cache
```

## Key Files

| File | Purpose |
|------|---------|
| `pixi.toml` | Project manifest (deps, tasks, channels, platforms) |
| `pyproject.toml` | Alternative manifest (Python projects) |
| `pixi.lock` | Lockfile — commit to VCS |
| `.pixi/` | Local env directory — add to `.gitignore` |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PIXI_HOME` | Override global install location (default `~/.pixi`) |
| `PIXI_COLOR` | Force color output: `always`, `never`, `auto` |
| `PIXI_NO_PROGRESS` | Disable progress bars |
| `PIXI_FROZEN` | Equivalent to `--frozen` |
| `PIXI_LOCKED` | Equivalent to `--locked` |