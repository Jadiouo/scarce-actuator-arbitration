"""GPU (torch, float64) accelerators for the two bottlenecks of the project.

dp_torch  : batched relative value iteration of the full-information semi-MDP (arbitration/dp.py)
sim_torch : batched, seed-exact simulator equivalent to arbitration.model.run

Both fall back to CPU torch when CUDA is unavailable. Nothing here is imported by the
rest of the package, so importing `arbitration` never requires torch.
"""


def default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
