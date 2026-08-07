#!/bin/bash
set -euo pipefail

repository_directory="${1:-$HOME/biogeme}"
git_reference="${2:-main}"
repository_url="${BIOGEME_REPOSITORY_URL:-https://github.com/michelbierlaire/biogeme.git}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is not available on this host." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it in your user account, then run this script again." >&2
    echo "See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [[ ! -d "$repository_directory/.git" ]]; then
    git clone "$repository_url" "$repository_directory"
fi

git -C "$repository_directory" fetch --prune origin
git -C "$repository_directory" checkout "$git_reference"

if git -C "$repository_directory" show-ref --verify --quiet "refs/remotes/origin/$git_reference"; then
    git -C "$repository_directory" merge --ff-only "origin/$git_reference"
fi

(
    cd "$repository_directory"
    uv sync --frozen
)

echo "Biogeme repository: $repository_directory"
echo "Git commit: $(git -C "$repository_directory" rev-parse HEAD)"
echo "Python: $repository_directory/.venv/bin/python"
