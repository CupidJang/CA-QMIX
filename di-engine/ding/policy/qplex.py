"""QPLEX policy: the QMIX policy with the QPLEX model.

Inherits QMIXPolicy so the training procedure (including the target hidden-state reset) matches QMIX.
"""
from typing import List, Tuple

from ding.utils import POLICY_REGISTRY
from .qmix import QMIXPolicy


@POLICY_REGISTRY.register('qplex')
class QPLEXPolicy(QMIXPolicy):
    config = dict(QMIXPolicy.config)
    config['type'] = 'qplex'

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qplex', ['ding.model.template.qplex']
