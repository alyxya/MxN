"""Generation API adapter; all learning/inference math comes from the snapshot."""
import torch
from .matrix_network import MatrixNetwork as HistoricalMatrixNetwork

class MatrixNetwork(HistoricalMatrixNetwork):
    def __init__(self, *, n, vocab, eos_token, device=None):
        super().__init__(n=n, vocab=vocab, device=device)
        self.eos_token = eos_token
        self.eos_id = self.stoi[eos_token]

    @torch.no_grad()
    def generate(self, prefix, max_len):
        self.reset_state()
        self.apply_context(self.encode(prefix))
        out = []
        for _ in range(max_len):
            token = self.predict()
            if token == self.eos_id:
                return "".join(out), True
            out.append(self.decode(token))
            self.apply_context([token])
        return "".join(out), False
