# activate.sh - activate the project venv in Git Bash
# Usage: source activate.sh      (or: . activate.sh)
# Undo with: deactivate

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Run this with: source activate.sh"
    exit 1
fi

# The venv's own Scripts/activate has the old project path (E:\Prog\summermute) baked in,
# so set the environment up here relative to this file instead.
_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_venv_dir="$_project_dir/.venv313"

if [ ! -x "$_venv_dir/Scripts/python.exe" ]; then
    echo "❌ venv not found: $_venv_dir"
    unset _project_dir _venv_dir
    return 1
fi

if type deactivate >/dev/null 2>&1; then
    deactivate
fi

_OLD_PATH="$PATH"
_OLD_PS1="$PS1"
export VIRTUAL_ENV="$_venv_dir"
export PATH="$_venv_dir/Scripts:$PATH"     # venv first, ahead of Cygwin's Python 2.7
export PYTHONUTF8=1                        # print ✅ and õäöü without encoding errors
export PYTHONUNBUFFERED=1                  # show prompts and progress immediately in mintty
PS1="(sumerute) $PS1"
hash -r

deactivate () {
    export PATH="$_OLD_PATH"
    PS1="$_OLD_PS1"
    unset VIRTUAL_ENV PYTHONUTF8 PYTHONUNBUFFERED _OLD_PATH _OLD_PS1
    unset -f deactivate
    hash -r
}

echo "✅ Activated $(python --version) from $_venv_dir"
unset _project_dir _venv_dir
