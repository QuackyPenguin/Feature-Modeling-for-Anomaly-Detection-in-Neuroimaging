# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist


# ---------------------------------------------------------------------
#  Memory‑bank (queue) -------------------------------------------------
# ---------------------------------------------------------------------
class MemoryBank(nn.Module):
    """
    FIFO queue that stores the last `bank_size` L2‑normalised feature vectors.

    Arguments
    ---------
    bank_size: int
        Maximum number of vectors stored in the bank.
    feat_dim:  int
        Dimensionality of each feature vector.
    momentum:  float in [0,1] or None
        If given, the bank is updated with
               v_new = m * v_old + (1-m) * v_in          (MoCo style)
        otherwise vectors are overwritten (FIFO queue).
    """

    def __init__(self,
                 bank_size: int = 1024,
                 feat_dim:  int = 128,
                 momentum:   float | None = None,
                 device: str = "cuda"):
        super().__init__()

        self.register_buffer("bank", torch.empty(bank_size, feat_dim, device=device))
        self.register_buffer("ptr",  torch.zeros((), dtype=torch.long))
        self.register_buffer("valid", torch.zeros((), dtype=torch.long))  # how many slots are filled
        self.momentum = momentum

    # -----------------------------------------------------------------
    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """Enqueue a mini‑batch of *already* normalised features."""
        feats = F.normalize(feats, dim=1)              # (B,D)
        B, _ = feats.shape
        K, ptr = int(self.bank.size(0)), int(self.ptr)

        n_insert = min(B, K)                           # just in case

        end = ptr + n_insert
        if end <= K:                                   # contiguous
            self._insert(ptr, end, feats[:n_insert])
        else:                                          # wrap‑around
            first = K - ptr
            self._insert(ptr,   K, feats[:first])
            self._insert(0, end % K, feats[first:n_insert])

        self.ptr   = torch.as_tensor(end % K, device=self.bank.device)
        self.valid = torch.clamp(self.valid + n_insert, max=K)

    @torch.no_grad()
    def _insert(self, start, end, vecs):
        if self.momentum is None or self.valid == 0:
            self.bank[start:end] = vecs
        else:
            # MoCo‑style momentum update
            old = self.bank[start:end]
            self.bank[start:end] = F.normalize(
                old * self.momentum + vecs * (1.0 - self.momentum), dim=1)

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        """Return the *filled* part of the bank (tensor view, no copy)."""
        return self.bank[: int(self.valid)]


# ---------------------------------------------------------------------
#  NT‑Xent loss with external memory‑bank ------------------------------
# ---------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """
    SimCLR NT‑Xent loss with an external (queue) memory‑bank.

    Arguments
    ---------
    bank          : MemoryBank         – queue that stores negatives
    temperature   : float              – soft‑max temperature (default 0.07)
    gather_distrib: bool               – gather mini‑batch across GPUs
    """

    def __init__(self,
                 bank: MemoryBank,
                 temperature: float = 0.07,
                 gather_distrib: bool = False):
        super().__init__()
        self.bank = bank
        self.T    = temperature
        self.gather_distrib = gather_distrib

    # -----------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor (2N, D)
            Concatenation of projections for two stochastic views of a
            mini‑batch.  First N rows = view 1 , next N rows = view 2.
        """
        z = F.normalize(z, dim=1)
        if self.gather_distrib and dist.is_initialized():
            z = self._gather_embeddings(z)        # (2N*world, D)

        N = z.size(0) // 2
        v1, v2   = z[:N], z[N:]                  # (N,D) each
        anchors  = torch.cat([v1, v2], dim=0)    # (2N,D)
        positives= torch.cat([v2, v1], dim=0)    # (2N,D)

        # positive similarities
        pos_sim  = (anchors * positives).sum(dim=1, keepdim=True)  # (2N,1)

        # in‑batch negatives
        sim_mat  = torch.matmul(anchors, anchors.T)                # (2N,2N)
        mask     = torch.eye(2*N, dtype=torch.bool,
                             device=anchors.device)
        mask |= mask.roll(N, dims=0)
        batch_neg= sim_mat.masked_fill(mask, -1e9)
        batch_neg= batch_neg[~mask].view(2*N, -1)                  # (2N, 2N-2)

        # memory‑bank negatives
        bank = self.bank.get()                                     # (M,D)
        if bank.numel() == 0:
            neg_sim = anchors.new_empty(anchors.size(0), 0)
        else:
            neg_sim = torch.matmul(anchors, bank.T)                # (2N,M)

        logits = torch.cat([pos_sim, batch_neg, neg_sim], dim=1) / self.T
        log_p  = F.log_softmax(logits, dim=1)
        loss   = -(log_p[:, 0]).mean()

        # -----------------------------------------------------------------
        # enqueue **after** the gradient step finishes (call from training loop)
        return loss

    # helper ----------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _gather_embeddings(z):
        world = dist.get_world_size()
        if world == 1:
            return z
        zs  = [torch.empty_like(z) for _ in range(world)]
        dist.all_gather(zs, z.contiguous())
        return torch.cat(zs, dim=0)
