"""Dual-stream generator; the residual magnitude head and atan2 phase head follow AP-BWE (Lu et al., 2024) and the Lattice coupling follows CIS-BWE (Tamiti et al., ACL 2026); the trunk, bridge, prior, graft and quantum-inspired modules are ours."""
import math

import torch
import torch.nn as nn

from bwe_sb.losses.phase import antiwrap, frame_f0
from bwe_sb.models.bridge import t_embed as sb_t_embed

def phase_derivatives(pha):
    """Instantaneous frequency and group delay of a wrapped phase spectrogram."""
    d_t = antiwrap(pha - torch.roll(pha, 1, dims=-1))
    d_f = antiwrap(pha - torch.roll(pha, 1, dims=-2))
    d_t[..., 0] = 0.0
    d_f[..., 0, :] = 0.0
    return d_t, d_f

def baseband_phase_diff(pha, hop, n_fft):
    """Baseband phase difference (BPD), from Gaznepoglu et al., IWAENC 2026."""
    F = pha.shape[-2]
    m = torch.arange(F, device=pha.device, dtype=pha.dtype).view(1, -1, 1)
    tpd = antiwrap(pha - torch.roll(pha, 1, dims=-1))
    bpd = antiwrap(tpd - 2.0 * torch.pi * m * hop / n_fft)
    bpd[..., 0] = 0.0
    return bpd

def graft_mask(n_bins, cut_bin, width, device, dtype):
    """Raised-cosine crossover: 0 below the cutoff, 1 above it."""
    f = torch.arange(n_bins, device=device, dtype=dtype).view(1, -1, 1)
    if not torch.is_tensor(cut_bin):
        cut_bin = torch.full((1,), float(cut_bin), device=device, dtype=dtype)
    cb = cut_bin.reshape(-1, 1, 1).to(dtype)
    width = max(1.0, float(width))
    x = ((f - cb) / width).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * x)

def sbr_prior(mag_nb, cut_bin, decay=None):
    """Coarse wideband log-magnitude prior by spectral band replication."""
    B, F, T = mag_nb.shape
    if not torch.is_tensor(cut_bin):
        cut_bin = torch.full((B,), int(cut_bin), device=mag_nb.device, dtype=torch.long)
    cut_bin = cut_bin.clamp(4, F - 1)
    idx = torch.arange(F, device=mag_nb.device).view(1, -1).expand(B, -1)
    c = cut_bin.view(-1, 1)
    span = (c - 1).clamp(min=1)
    src = torch.where(idx < c, idx, (c - 1 - ((idx - c) % span)).clamp(0, F - 1))
    out = torch.gather(mag_nb, 1, src.unsqueeze(-1).expand(-1, -1, T))
    above = (idx >= c).unsqueeze(-1).to(mag_nb.dtype)
    if decay is not None:
        out = out + decay.view(1, -1, 1) * above
    return out * above + mag_nb * (1 - above)

def mp_add(a, b, tau):
    """Magnitude-preserving mix (Richter et al., ICASSP 2026)."""
    t = tau.view(1, -1, 1)
    return ((1 - t) * a + t * b) / ((1 - t) ** 2 + t ** 2).sqrt()

class Phase2DHead(nn.Module):
    """A 2D refinement of the phase prediction, on the time-frequency plane."""
    def __init__(self, ch=16, dils=(1, 4, 16), n_in=4):
        super().__init__()
        self.inp = nn.Conv2d(n_in, ch, 3, padding=1)
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, (5, 3), padding=(2 * d, 1), dilation=(d, 1))
            for d in dils])
        self.mix = nn.Conv2d(ch * len(dils), ch, 1)
        self.act = nn.GELU()
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.normal_(self.out.weight, std=1e-4)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats):
        x = self.act(self.inp(feats))
        x = self.act(self.mix(torch.cat([b(x) for b in self.branches], dim=1)))
        return self.out(x).squeeze(1)

def harmonic_phase_init(pha_nb, f0, df, n_bins):
    """Physics-derived initial wideband phase, from the harmonic relation."""
    B, Fb, T = pha_nb.shape
    T = min(T, f0.shape[-1])
    f0 = f0[..., :T].clamp(min=1e-3)
    k1 = (f0 / df).round().long().clamp(1, Fb - 1)
    phi1 = torch.gather(pha_nb[..., :T], 1, k1.unsqueeze(1))
    k = torch.arange(Fb, device=pha_nb.device).view(1, -1, 1).float()
    h = (k * df / f0.unsqueeze(1)).round().clamp(min=1.0)
    init = antiwrap(h * phi1)
    if T < pha_nb.shape[-1]:
        init = torch.cat([init, init[..., -1:].expand(-1, -1, pha_nb.shape[-1] - T)], -1)
    return init

def _istft_mag_pha(log_amp, pha, n_fft, hop, win, center=True):
    w = torch.hann_window(win, device=log_amp.device)
    spec = torch.complex(log_amp.exp() * pha.cos(), log_amp.exp() * pha.sin())
    return torch.istft(spec, n_fft, hop_length=hop, win_length=win, window=w, center=center)

class Lattice1d(nn.Module):
    def __init__(self, dim, amp_block, pha_block, share=False, gate_init=0.1,
                 mp_add=False, adaln=False):
        super().__init__()
        self.use_mp, self.use_adaln = mp_add, adaln
        if mp_add:
            self.tau_m = nn.Parameter(torch.full((dim,), -4.0))
            self.tau_p = nn.Parameter(torch.full((dim,), -4.0))
        if adaln:
            self.ln_m = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.ln_p = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.bm = amp_block()
        self.bp = self.bm if share else pha_block()
        if share:
            self.gm = nn.Parameter(torch.ones(dim));  self.sm = nn.Parameter(torch.zeros(dim))
            self.gp = nn.Parameter(torch.ones(dim));  self.sp = nn.Parameter(torch.zeros(dim))
        g0 = float(gate_init)
        self.a1 = nn.Parameter(torch.full((1,), g0))
        self.a2 = nn.Parameter(torch.full((1,), g0))
        self.b1 = nn.Parameter(torch.full((1,), g0))
        self.b2 = nn.Parameter(torch.full((1,), g0))

    def forward(self, m, p, y_m=None, y_p=None, mod=None):
        mi, pi = m + self.a1 * p, p + self.b1 * m
        if self.use_mp and y_m is not None:
            mi = mp_add(mi, y_m, torch.sigmoid(self.tau_m))
            pi = mp_add(pi, y_p, torch.sigmoid(self.tau_p))
        if self.use_adaln and mod is not None:
            gm, sm, hm, gp, sp, hp = mod
            mi = self.ln_m(mi.transpose(1, 2)).transpose(1, 2) * (1 + gm) + sm
            pi = self.ln_p(pi.transpose(1, 2)).transpose(1, 2) * (1 + gp) + sp
        elif self.bm is self.bp:
            mi = mi * self.gm.view(1, -1, 1) + self.sm.view(1, -1, 1)
            pi = pi * self.gp.view(1, -1, 1) + self.sp.view(1, -1, 1)
        m2, p2 = self.bm(mi), self.bp(pi)
        if self.use_adaln and mod is not None:
            m2, p2 = m2 * (1 + hm), p2 * (1 + hp)
        return m2 + self.a2 * p2, p2 + self.b2 * m2

class DualStreamBWE(nn.Module):
    def __init__(self, h):
        super().__init__()
        n_bins  = h.n_fft // 2 + 1
        dim     = int(getattr(h, "gen_dim", 512) or 512)
        lattice = int(getattr(h, "gen_lattice", 2) or 2)
        pk_m = int(getattr(h, "gen_pre_kernel_mag", None)
                   or getattr(h, "gen_pre_kernel", None) or 1)
        pk_p = int(getattr(h, "gen_pre_kernel_pha", None) or 7)
        self.pha_deriv = getattr(h, "gen_phase_deriv", False) or False
        self.pha_mode = str(getattr(h, "gen_phase_mode", None) or "scratch")
        self.sr = int(getattr(h, "hr_sampling_rate", 16000) or 16000)
        self.hop = int(getattr(h, "hop_size", 80) or 80)
        self.win = int(getattr(h, "win_size", 320) or 320)
        self.n_fft = int(h.n_fft)
        self.cond_dim = int(getattr(h, "gen_cond_dim", None) or 0)
        self.mag_fusion = bool(getattr(h, "gen_mag_fusion", False))
        self.pha_2d = bool(getattr(h, "gen_phase_2d", False))
        self.hf_prior = bool(getattr(h, "gen_hf_prior", False))
        self.mp_add = bool(getattr(h, "gen_mp_add", False))
        self.adaln = bool(getattr(h, "gen_adaln", False))
        self.sb = bool(getattr(h, "gen_sb", False))
        self.t_dim = int(getattr(h, "gen_t_dim", 64) or 64) if self.sb else 0
        self.bb = "selssm"
        self.amp_head = str(getattr(h, "gen_amp_head", None) or "linear")
        self.amp_qfeat = str(getattr(h, "gen_amp_qfeat", None) or "none")
        share   = bool(getattr(h, "gen_share_streams", False))
        gate    = float(getattr(h, "gen_gate_init", 0.1) or 0.1)
        _qdim = 0
        if self.amp_qfeat == "angle":
            from bwe_sb.models.backbones import AngleEmbedding
            self.qfeat = AngleEmbedding(n_bins, int(getattr(h, "gen_amp_qfeat_dim", 16) or 16))
            _qdim = self.qfeat.out_dim
        _n_in_m = n_bins * (2 if self.hf_prior else 1) + _qdim
        self.amp_in_mps = bool(getattr(h, "gen_amp_in_mps", False)) and pk_m == 1
        if self.amp_in_mps:
            from bwe_sb.models.backbones import MPSProj
            self.pre_m = MPSProj(_n_in_m, dim)
        else:
            self.pre_m = nn.Conv1d(_n_in_m, dim, pk_m, padding=pk_m // 2)
        self.graft = int(h.get('gen_graft', 0) or 0)
        self.default_cut_bin = max(4, int(round(
            (int(getattr(h, "lr_sampling_rate", 2000) or 2000) / 2)
            / (self.sr / 2) * (n_bins - 1))))
        if self.hf_prior:
            self.sbr_decay = nn.Parameter(torch.zeros(n_bins))
        _nch = {False: 1, True: 3, "only": 2}[self.pha_deriv]
        assert self.pha_mode in ("scratch", "residual", "harmonic"), self.pha_mode
        self.pre_p = nn.Conv1d(n_bins * _nch, dim, pk_p, padding=pk_p // 2)
        if self.pha_mode == "harmonic":
            self.phi_offset = nn.Parameter(torch.zeros(n_bins, 1))
        self.nm = nn.LayerNorm(dim, eps=1e-6)
        self.np = nn.LayerNorm(dim, eps=1e-6)
        from bwe_sb.models.backbones import SelectiveSSMStack
        amp_mk = lambda: SelectiveSSMStack(dim, depth=int(getattr(h, "gen_amp_depth", None) or 3))
        pha_mk = lambda: SelectiveSSMStack(dim, depth=int(getattr(h, "gen_pha_depth", None) or 3))
        self.blocks = nn.ModuleList([Lattice1d(dim, amp_mk, pha_mk, share, gate,
                                               mp_add=self.mp_add, adaln=self.adaln)
                                     for _ in range(lattice)])
        if self.mp_add:
            self.y_m = nn.Conv1d(n_bins, dim, 1)
            self.y_p = nn.Conv1d(2 * n_bins, dim, 1)
            self._zero_y = True
        self.film_in = self.cond_dim + self.t_dim
        self.n_mod = 6 if self.adaln else 4
        if self.film_in:
            self.film = nn.ModuleList([nn.Linear(self.film_in, self.n_mod * dim)
                                       for _ in range(lattice)])
        self.post_m = nn.LayerNorm(dim, eps=1e-6)
        self.post_p = nn.LayerNorm(dim, eps=1e-6)
        if self.amp_head == "mps":
            from bwe_sb.models.backbones import MPSHead
            self.out_m = MPSHead(dim, n_bins, rank=getattr(h, "gen_amp_mps_rank", None),
                                 cores=int(getattr(h, "gen_amp_mps_cores", 8) or 8))
        else:
            self.out_m = nn.Linear(dim, n_bins)
        self.out_r = nn.Linear(dim, n_bins)
        self.out_i = nn.Linear(dim, n_bins)
        if self.mag_fusion:
            self.fuse_a = nn.Parameter(torch.tensor(6.0))
            self.fuse_g = nn.Parameter(torch.tensor([-6.0]))
        if self.pha_2d:
            self.pha_head2d = Phase2DHead(
                ch=int(getattr(h, "gen_phase_2d_ch", 16) or 16), n_in=4)
        self.amp_head2d = Phase2DHead(ch=16, n_in=1)
        self.apply(self._init)
        if self.hf_prior:
            k = torch.arange(n_bins, dtype=torch.float32)
            cb = float(self.default_cut_bin)
            oct_ = torch.log2(k.clamp(min=cb) / cb)
            with torch.no_grad():
                self.sbr_decay.copy_(-6.0 * oct_ * math.log(10.0) / 20.0)
        if self.film_in:
            for _l in self.film:
                nn.init.zeros_(_l.weight)
                nn.init.zeros_(_l.bias)
        if getattr(self, "_zero_y", False):
            for _l in (self.y_m, self.y_p):
                nn.init.zeros_(_l.weight)
                nn.init.zeros_(_l.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def coarse_prior(self, mag_nb, cut_bin=None):
        """The SBR prior on its own, for use as the bridge's x0 endpoint."""
        if not self.hf_prior:
            return mag_nb
        cb = self.default_cut_bin if cut_bin is None else cut_bin
        return sbr_prior(mag_nb, cb, self.sbr_decay)

    def forward(self, mag_nb, pha_nb, cond=None, cut_bin=None, t=None,
                x0_mag=None, x0_pha=None):
        mag_in = mag_nb
        prior = None
        if self.hf_prior:
            cb = self.default_cut_bin if cut_bin is None else cut_bin
            prior = sbr_prior(mag_nb, cb, self.sbr_decay)
            mag_in = torch.cat([mag_nb, prior], dim=1)
        if self.amp_qfeat == "angle":
            mag_in = torch.cat([mag_in, self.qfeat(mag_nb)], dim=1)
        m = self.nm(self.pre_m(mag_in).transpose(1, 2)).transpose(1, 2)
        pin = pha_nb
        if self.pha_deriv:
            d_t, d_f = phase_derivatives(pha_nb)
            pin = torch.cat(([d_t, d_f] if self.pha_deriv == "only"
                             else [pha_nb, d_t, d_f]), dim=1)
        p = self.np(self.pre_p(pin).transpose(1, 2)).transpose(1, 2)
        ctx = None
        if self.film_in:
            parts = []
            if self.cond_dim:
                parts.append(cond if cond is not None else
                             mag_nb.new_zeros(mag_nb.shape[0], self.cond_dim))
            if self.t_dim:
                tv = t if t is not None else mag_nb.new_zeros(mag_nb.shape[0])
                if not torch.is_tensor(tv):
                    tv = mag_nb.new_full((mag_nb.shape[0],), float(tv))
                parts.append(sb_t_embed(tv.reshape(-1), self.t_dim))
            ctx = torch.cat(parts, dim=-1)
        y_m = y_p = None
        if self.mp_add:
            o_mag = x0_mag if x0_mag is not None else (prior if prior is not None else mag_nb)
            o_pha = x0_pha if x0_pha is not None else pha_nb
            d_t, d_f = phase_derivatives(o_pha)
            y_m = self.y_m(o_mag)
            y_p = self.y_p(torch.cat([d_t, d_f], dim=1))

        for i, blk in enumerate(self.blocks):
            mod = None
            if ctx is not None:
                ch = self.film[i](ctx).chunk(self.n_mod, dim=-1)
                if self.adaln:
                    gm, sm, hm, gp, sp, hp = (c.unsqueeze(-1) for c in ch)
                    mod = (gm, sm, hm, gp, sp, hp)
                else:
                    gm, sm, gp, sp = (c.unsqueeze(-1) for c in ch)
                    m = m * (1 + gm) + sm
                    p = p * (1 + gp) + sp
            m, p = blk(m, p, y_m=y_m, y_p=y_p, mod=mod)
        base_m = prior if (self.hf_prior and prior is not None) else mag_nb
        mag_wb = base_m + self.out_m(self.post_m(m.transpose(1, 2))).transpose(1, 2)
        pf = self.post_p(p.transpose(1, 2))
        r_hat = self.out_r(pf).transpose(1, 2)
        i_hat = self.out_i(pf).transpose(1, 2)
        d_pha = torch.atan2(i_hat, r_hat)
        if self.pha_mode == "residual":
            pha_wb = antiwrap(pha_nb + d_pha)
        elif self.pha_mode == "harmonic":
            with torch.no_grad():
                f0, _ = frame_f0(_istft_mag_pha(mag_nb, pha_nb, self.n_fft, self.hop,
                                                self.win), self.sr, self.hop, self.win)
            init = harmonic_phase_init(pha_nb, f0, self.sr / self.n_fft, pha_nb.shape[1])
            pha_wb = antiwrap(init + self.phi_offset.unsqueeze(0) + d_pha)
        else:
            pha_wb = d_pha

        if self.pha_2d:
            bpd_hat = baseband_phase_diff(pha_wb, self.hop, self.n_fft)
            bpd_in = baseband_phase_diff(pha_nb, self.hop, self.n_fft)
            feats = torch.stack([pha_wb.sin(), pha_wb.cos(), bpd_hat - bpd_in, mag_wb],
                                dim=1)
            pha_wb = antiwrap(pha_wb + self.pha_head2d(feats))

        if self.mag_fusion:
            a = torch.sigmoid(self.fuse_a)
            amp_m = mag_wb.exp()
            amp_p = self.fuse_g.exp() * torch.sqrt(r_hat ** 2 + i_hat ** 2 + 1e-12)
            mag_wb = torch.log(a * amp_m + (1 - a) * amp_p + 1e-5)

        mag_wb = mag_wb + self.amp_head2d(torch.tanh(mag_wb / 5.0).unsqueeze(1))

        if self.graft:
            cb = self.default_cut_bin if cut_bin is None else cut_bin
            w = graft_mask(mag_wb.shape[1], cb, self.graft, mag_wb.device, mag_wb.dtype)
            mag_wb = (1 - w) * mag_nb + w * mag_wb
            cs = (1 - w) * pha_nb.cos() + w * pha_wb.cos()
            sn = (1 - w) * pha_nb.sin() + w * pha_wb.sin()
            pha_wb = torch.atan2(sn, cs)

        com_wb = torch.stack((mag_wb.exp() * pha_wb.cos(), mag_wb.exp() * pha_wb.sin()), dim=-1)
        return mag_wb, pha_wb, com_wb
