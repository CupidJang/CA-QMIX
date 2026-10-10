# Modified for CA-QMIX: registers the CA-QMIX and CollaQ controllers; unused controllers removed.
REGISTRY = {}

from .basic_controller import BasicMAC
from .caq_controller import CaqMAC
from .colla_controller import CollaMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["caq_mac"] = CaqMAC
REGISTRY["colla_mac"] = CollaMAC
