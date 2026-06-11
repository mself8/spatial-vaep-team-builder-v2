"""VAEP pipeline package (raw events -> SPADL -> VAEP).

Importing this package puts the vendored ``vaep/lib`` directory on
``sys.path`` so that the bundled ``datatools`` library (which uses
absolute internal imports such as ``from datatools.loaders... import``)
can be imported from anywhere in the repository.
"""
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
