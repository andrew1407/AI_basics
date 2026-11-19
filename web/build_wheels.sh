#!/bin/sh
# neurolab 0.3.5 (4-lab) is published only as a source archive and predates NumPy 2, so Pyodide can't install it.
# Build a pure-Python wheel into web/wheels/, with a shim in neurolab/__init__.py for the NumPy 1 aliases it uses.
set -e
cd "$(dirname "$0")"
tmp=$(mktemp -d)
python3 -m pip download neurolab==0.3.5 --no-deps --no-binary :all: -d "$tmp" -q
tar xzf "$tmp/neurolab-0.3.5.tar.gz" -C "$tmp"
init="$tmp/neurolab-0.3.5/neurolab/__init__.py"
{ cat <<'PY'
import numpy as _np  # NumPy 2 removed these aliases
for _name, _value in dict(Inf=_np.inf, row_stack=_np.vstack,
                          asfarray=lambda a, dtype=_np.float64: _np.asarray(a, dtype=dtype)).items():
    if not hasattr(_np, _name): setattr(_np, _name, _value)
PY
sed '1s/^\xEF\xBB\xBF//' "$init"; } > "$tmp/init.py" && mv "$tmp/init.py" "$init"
rm -rf wheels && mkdir wheels
python3 -m pip wheel --no-deps -q -w wheels "$tmp/neurolab-0.3.5" 2>/dev/null
rm -rf "$tmp"
ls wheels
