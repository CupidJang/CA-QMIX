"""Qatten policy: the QMIX policy with the Qatten model.

Training procedure and hyperparameters match QMIX; only the mixer differs.
"""
from typing import List, Tuple

from ding.utils import POLICY_REGISTRY
from .qmix import QMIXPolicy


@POLICY_REGISTRY.register('qatten')
class QattenPolicy(QMIXPolicy):
    config = dict(QMIXPolicy.config)
    config['type'] = 'qatten'

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qatten', ['ding.model.template.qatten']
