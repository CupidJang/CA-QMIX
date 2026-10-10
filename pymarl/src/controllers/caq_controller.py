# MAC for CA-QMIX: identical to BasicMAC except forward() can also return the base-stream Q,
# which the learner needs for the DeltaQ regulariser. Action selection uses adjusted Q only.
from .basic_controller import BasicMAC


class CaqMAC(BasicMAC):

    def forward(self, ep_batch, t, test_mode=False, return_base=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if test_mode:
            self.agent.eval()
        if return_base:
            q, q_base, self.hidden_states = self.agent(agent_inputs, self.hidden_states, return_base=True)
            b = ep_batch.batch_size
            return q.view(b, self.n_agents, -1), q_base.view(b, self.n_agents, -1)
        q, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return q.view(ep_batch.batch_size, self.n_agents, -1)
