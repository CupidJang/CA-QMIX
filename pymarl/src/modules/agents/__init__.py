# Modified for CA-QMIX: registers the CA-QMIX and CollaQ agents; unused agents removed.
REGISTRY = {}

from .rnn_agent import RNNAgent
from .caqmix_agent import CaqMixAgent
from .collaq_agent import CollaQAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["caqmix"] = CaqMixAgent
REGISTRY["collaq"] = CollaQAgent
