# Modified for CA-QMIX: registers the CA-QMIX and CollaQ learners; unused learners removed.
from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .collaq_learner import CollaQLearner
from .caqmix_learner import CaqMixLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["collaq_learner"] = CollaQLearner
REGISTRY["caqmix_learner"] = CaqMixLearner
