"""REQ-OBS-10: independent obs reconstructor.  Must NOT import the simulator or the feature builder (T-56 checks the file
statically); written only from spec s3.1 / s4.4 and fed only the public log."""
import math
from typing import Any, Dict, Optional

import torch

_PUBLIC = ("z1", "u1", "v1", "won1", "susp1", "consts")
_LAMBDA = 0.99
_K = 8


def reconstruct(public_log: Dict[str, Any], version: str = "public", extra: Optional[Dict[str, Any]] = None) -> Any:
    """Rebuild obs[...,T,F] from the public log {z1,u1,v1,won1,susp1,consts}.  Unknown (private) keys -> ValueError.
    version 'announced' needs extra={'S':..., 'audited_last':...}; 'oracle' needs extra={'u0':..., 'u2':...}."""
    unknown = set(public_log) - set(_PUBLIC)
    missing = set(_PUBLIC) - set(public_log)
    if unknown:
        raise ValueError(f"non-public keys in the log: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing log keys: {sorted(missing)}")
    if version not in ("public", "announced", "oracle"):
        raise ValueError(version)
    need = {"public": (), "announced": ("S", "audited_last"), "oracle": ("u0", "u2")}[version]
    extra = extra or {}
    if set(extra) != set(need):
        raise ValueError(f"version {version} needs extra keys {need}, got {sorted(extra)}")
    c = public_log["consts"]
    r, L = float(c["r"]), float(c["L"])
    sd = math.sqrt(1.0 - r * r)
    z1 = public_log["z1"]
    lead, T = z1.shape[:-1], z1.shape[-1]
    dt, dev = z1.dtype, z1.device

    def tb(x):                       # [..., T] -> [T, B] contiguous
        return x.reshape(-1, T).to(dt).t().contiguous()

    z, u, v, won, susp = (tb(public_log[k]) for k in ("z1", "u1", "v1", "won1", "susp1"))
    ext = {k: tb(extra[k]) for k in need}
    B = z.shape[1]
    zeros = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
    ema_res, ema_d, ema_win = zeros(B), zeros(B), zeros(B)
    last_notice = torch.full((B,), -math.inf, dtype=dt, device=dev)
    seen = zeros(B)
    won_h, d_h = zeros(B, _K), zeros(B, _K)
    v_prev = zeros(B)
    out = []
    one_m = 1.0 - _LAMBDA
    for i in range(T):
        t = float(i + 1)
        s1 = susp[i]
        last_notice = torch.where(s1 != seen, torch.full_like(last_notice, t), last_notice)
        seen = s1.clone()
        since = t - last_notice
        smt = s1 - t
        f01 = torch.clamp(z[i] / 3.0, -1.0, 1.0)
        f02 = 2.0 * u[i] - 1.0
        f03 = (smt < 0).to(dt)
        f04 = torch.clamp(smt, min=0.0) / L
        f05 = torch.clamp(since, max=2.0 * L) / (2.0 * L)
        zv = torch.special.ndtri(torch.clamp(v_prev, 1e-12, 1 - 1e-12))
        zw = torch.special.ndtri(torch.clamp(u[i], 1e-12, 1 - 1e-12))
        res = (zw - r * zv) / sd
        f22 = torch.where(won_h[:, 0] > 0.5, torch.clamp(res / 6.0, -1.0, 1.0), torch.zeros_like(res))
        ema_res = _LAMBDA * ema_res + one_m * f22
        ema_d = _LAMBDA * ema_d + one_m * d_h[:, 0]
        ema_win = _LAMBDA * ema_win + one_m * won_h[:, 0]
        parts = [torch.stack([f01, f02, f03, f04, f05], -1), won_h, d_h, torch.stack([f22, ema_res, ema_d, ema_win], -1)]
        if version == "announced":
            parts.append(torch.stack([ext["S"][i], ext["audited_last"][i]], -1))
        elif version == "oracle":
            parts.append(torch.stack([2.0 * ext["u0"][i] - 1.0, 2.0 * ext["u2"][i] - 1.0], -1))
        out.append(torch.cat(parts, -1))
        # end of round: shift own history
        won_h = torch.cat([won[i][:, None], won_h[:, :-1]], 1)
        d_h = torch.cat([(v[i] - u[i])[:, None], d_h[:, :-1]], 1)
        v_prev = v[i].clone()
    o = torch.stack(out, 1)
    return o.reshape(*lead, T, o.shape[-1])
