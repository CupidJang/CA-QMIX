# Modified for CA-QMIX: unused run modes removed.
from .run import run as default_run

REGISTRY = {}
REGISTRY["default"] = default_run
