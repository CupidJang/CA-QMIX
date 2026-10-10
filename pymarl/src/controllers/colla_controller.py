# MAC for CollaQ: BasicMAC with a 3-stream hidden state and optional Q_colla_alone output
# (needed by the MARA regulariser in the learner).
from .basic_controller import BasicMAC


class CollaMAC(BasicMAC):

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1, -1)

    def forward(self, ep_batch, t, test_mode=False, return_extras=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if test_mode:
            self.agent.eval()
        if return_extras:
            q, q_ca, self.hidden_states = self.agent(agent_inputs, self.hidden_states, return_extras=True)
            b = ep_batch.batch_size
            return q.view(b, self.n_agents, -1), q_ca.view(b, self.n_agents, -1)
        q, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return q.view(ep_batch.batch_size, self.n_agents, -1)
