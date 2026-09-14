"""Report 17 figures, drawn from the .npy dumps written by sensitivity_maps.py.

Kept separate from the computation so the figures can be re-drawn without
repeating the rollouts. `sensitivity_maps.py` calls make_figures() at the end;
running this file directly re-draws from whatever is already in the output
directory:

    python figures.py [OUT_DIR]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

FIELD_KIND = {
    "temp": "T", "salt": "T", "prho": "T", "Nsqr": "T", "tke": "T",
    "dTdz": "W", "dSdz": "W", "drhodz": "W", "mld": "S",
}

# Observable names. A bare name is the full 3D field; the suffixes select a
# sub-observable, all of them derivable from the final-step dumps because both
# operations are linear (a depth average commutes with the tangent):
#
#   <field>_surf     surface level only
#   <field>_topN     restricted to the top N levels (still one value per cell)
#   <field>_avgN     averaged over the top N levels (one value per column)
#   <field>_absavgN  mean of |field| over the top N levels (one value per column)
#   <field>_rmsN     sqrt(mean of field^2) over the top N levels
#
# Levels are ordered deepest-first, so "top N" is the last N indices. Averages are
# unweighted over levels, not thickness-weighted.
#
# _absavgN and _rmsN are nonlinear, so their tangents are not the average of the
# field's tangents -- but they are still exact from the final-step dumps by the
# chain rule, since both reductions are pointwise-then-averaged:
#
#     d/dtheta mean|g|          = mean( sign(g) * dg/dtheta )
#     d/dtheta sqrt(mean g^2)   = mean( g * dg/dtheta ) / sqrt(mean g^2)
#
# |.| is not differentiable at g = 0; no cell in this run sits there, and the
# subgradient sign(0) = 0 is used if one ever does.

ORDER = [
    "temp_surf", "temp", "salt_surf", "salt", "prho_surf", "prho",
    "mld", "dTdz", "dSdz", "drhodz", "Nsqr_surf", "Nsqr", "tke_surf", "tke",
]

# Sub-observables added to answer "what about only the top few layers?"
DERIVED = [
    "dTdz_top3", "dTdz_top5", "dSdz_top3", "dSdz_top5", "drhodz_top5",
    "temp_avg3", "salt_avg3", "prho_avg3", "prho_avg5", "prho_top3",
    # 2D stratification metrics: one number per column, mappable
    "dTdz_avg3", "dTdz_absavg3", "dTdz_absavg5", "dTdz_rms3",
    "dSdz_absavg3", "drhodz_absavg3", "drhodz_rms3", "Nsqr_avg3", "Nsqr_absavg3",
]

ALL_OBS = ORDER + DERIVED

TS_FIELDS = ["temp_surf", "temp", "salt", "mld", "dTdz", "Nsqr", "tke"]


def _load(out_dir):
    d = {}
    d["xt"] = np.load(os.path.join(out_dir, "xt.npy"))
    d["yt"] = np.load(os.path.join(out_dir, "yt.npy"))
    d["zt"] = np.load(os.path.join(out_dir, "zt.npy"))
    d["maskT"] = np.load(os.path.join(out_dir, "maskT.npy"))
    d["summary"] = np.load(os.path.join(out_dir, "summary.npy"), allow_pickle=True)[0]
    d["timeseries"] = np.load(os.path.join(out_dir, "timeseries.npy"), allow_pickle=True)[0]
    meta = np.load(os.path.join(out_dir, "meta.npy"), allow_pickle=True)[0]
    d.update(meta)
    d["S"] = {}
    for name in d["points"]:
        d["S"][name] = {}
        for key in ("S_k", "S_e", "primal"):
            d["S"][name][key] = {
                f: np.load(os.path.join(out_dir, f"{name}_{key}_{f}_final.npy"))
                for f in FIELD_KIND
            }
        d["S"][name]["mld_mask"] = np.load(os.path.join(out_dir, f"{name}_mld_mask.npy"))
    return d


def _gram(sk, se, mask):
    """2x2 Gauss-Newton matrix of the log-sensitivities over `mask`."""
    a = float(np.sum(sk[mask] ** 2))
    b = float(np.sum(sk[mask] * se[mask]))
    c = float(np.sum(se[mask] ** 2))
    tr, det = a + c, a * c - b * b
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    lmax, lmin = tr / 2.0 + disc, tr / 2.0 - disc
    r = b / (a * c) ** 0.5 if a > 0 and c > 0 else float("nan")
    cond = lmax / lmin if lmin > 0 else float("inf")
    if abs(b) > 0:
        v = np.array([b, lmin - a])
    else:
        v = np.array([0.0, 1.0]) if a >= c else np.array([1.0, 0.0])
    v = v / (np.linalg.norm(v) + 1e-300)
    return dict(a=a, b=b, c=c, lmax=lmax, lmin=lmin, r=r, cond=cond, v_min=v)


def _rel_sens(sig, y, mask):
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    sd = float(np.std(y[mask]))
    if sd == 0:
        return float("nan")
    return float(np.sqrt(np.sum(sig[mask] ** 2) / n) / sd)


# --- joint-loss conditioning ---------------------------------------------------
# A weighted least-squares loss summing several observables, each normalised by
# its own variance so the terms are comparable, has a Gauss-Newton Hessian equal
# to the sum of the per-field 2x2 Grams. That sum is what a joint loss should be
# designed to condition, so it is computed here rather than in the rollout script
# -- it needs only the final-step dumps, so combinations can be added and the
# figures redrawn in seconds.

COMBOS = [
    ("temp_surf",),
    ("salt_surf",),
    ("mld",),
    ("Nsqr",),
    ("dTdz",),
    ("dSdz",),
    ("temp_surf", "salt_surf"),
    ("temp_surf", "mld"),
    ("temp_surf", "Nsqr"),
    ("temp_surf", "dTdz"),
    ("temp_surf", "dTdz", "dSdz"),
    ("temp_surf", "salt_surf", "dTdz", "dSdz"),
    ("temp_surf", "salt_surf", "mld"),
    ("dTdz_top5",),
    ("dSdz_top5",),
    ("prho_avg3",),
    ("temp_surf", "prho_avg3"),
    ("temp_surf", "dTdz_top5"),
    ("temp_surf", "dTdz_top5", "dSdz_top5"),
    ("temp_surf", "salt_surf", "dTdz_top5", "dSdz_top5"),
    ("dTdz_top3",),
    ("dSdz_top3",),
    ("temp_surf", "dTdz_top3", "dSdz_top3"),
    ("prho_top3",),
    ("temp_surf", "prho_top3"),
    ("dTdz_avg3",),
    ("dTdz_absavg3",),
    ("dTdz_rms3",),
    ("drhodz_absavg3",),
    ("Nsqr_absavg3",),
    ("temp_surf", "dTdz_absavg3"),
    ("temp_surf", "dTdz_absavg3", "dSdz_absavg3"),
    ("temp_surf", "salt_surf", "dTdz_absavg3", "dSdz_absavg3"),
]


def _gram_stats(H):
    a, b, c = H[0, 0], H[0, 1], H[1, 1]
    tr, det = a + c, a * c - b * b
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    lmax, lmin = tr / 2.0 + disc, tr / 2.0 - disc
    r = b / (a * c) ** 0.5 if a > 0 and c > 0 else float("nan")
    return lmin, (lmax / lmin if lmin > 0 else float("inf")), r


def compute_combo_stats(d, masks):
    """Per-combination lambda_min / cond / r of the variance-normalised joint Gram."""
    stats = {}
    for name in d["points"]:
        S = d["S"][name]
        per_field = {}
        for field in sorted({f for combo in COMBOS for f in combo}):
            m, sk, se, y = masks(d, S, field)
            sd = float(np.std(y[m])) or 1.0
            n = max(int(m.sum()), 1)
            scale = 1.0 / (n * sd * sd)
            a = float(np.sum(sk[m] ** 2)) * scale
            b = float(np.sum(sk[m] * se[m])) * scale
            c = float(np.sum(se[m] ** 2)) * scale
            per_field[field] = np.array([[a, b], [b, c]])
        rows = {}
        for combo in COMBOS:
            lmin, cond, r = _gram_stats(sum(per_field[f] for f in combo))
            rows[combo] = dict(lmin=lmin, cond=cond, r=r)
        stats[name] = rows
    return stats


def make_figures(out_dir):
    d = _load(out_dir)
    XT, YT, ZT_NP, MASK_NP = d["xt"], d["yt"], d["zt"], d["maskT"]
    NZ = ZT_NP.size
    SURFACE = NZ - 1
    MASK_W = MASK_NP[:, :, 1:] & MASK_NP[:, :, :-1]
    N_STEPS = d["n_steps"]
    summary, timeseries = d["summary"], d["timeseries"]
    points = d["points"]
    REF = points[0]
    ck_ref, ce_ref = d["point_params"][REF]
    S = d["S"][REF]

    def field_mask(field, mld_mask):
        kind = FIELD_KIND[field]
        return MASK_NP if kind == "T" else (MASK_W if kind == "W" else mld_mask)

    def masked(arr, m):
        return np.where(m, arr, np.nan)

    def masks(_d, S_, name):
        """(cell mask, S_k, S_e, primal) for a named observable.

        Handles the _surf / _topN / _avgN suffixes. Both restriction and depth
        averaging are linear, so they apply identically to the primal and to the
        tangents -- no rollout needed to add a sub-observable.
        """
        base, _, suffix = name.partition("_")
        if base not in FIELD_KIND:
            raise KeyError(name)
        m = field_mask(base, S_["mld_mask"])
        sk, se, y = S_["S_k"][base], S_["S_e"][base], S_["primal"][base]

        if not suffix:
            return m, sk, se, y
        if suffix == "surf":
            return m[:, :, SURFACE], sk[:, :, SURFACE], se[:, :, SURFACE], y[:, :, SURFACE]
        n = int("".join(ch for ch in suffix if ch.isdigit()))
        sl = slice(-n, None)  # levels are deepest-first, so the top N are the last N
        if suffix.startswith("top"):
            return m[:, :, sl], sk[:, :, sl], se[:, :, sl], y[:, :, sl]

        # a column enters only if every reduced level is ocean
        cm = m[:, :, sl].all(axis=2)
        ys, sks, ses = y[:, :, sl], sk[:, :, sl], se[:, :, sl]

        if suffix.startswith("avg"):
            return cm, sks.mean(axis=2), ses.mean(axis=2), ys.mean(axis=2)
        if suffix.startswith("absavg"):
            sgn = np.sign(ys)
            return cm, (sgn * sks).mean(axis=2), (sgn * ses).mean(axis=2), np.abs(ys).mean(axis=2)
        if suffix.startswith("rms"):
            ms = (ys ** 2).mean(axis=2)
            root = np.sqrt(ms)
            safe = np.where(root > 0, root, 1.0)
            return (cm & (root > 0), (ys * sks).mean(axis=2) / safe,
                    (ys * ses).mean(axis=2) / safe, root)
        raise KeyError(name)

    # The rollout script's summary only covers the base observables; the derived
    # ones are added here.
    for name in points:
        S_ = d["S"][name]
        for obs in DERIVED:
            m, sk, se, y = masks(d, S_, obs)
            g = _gram(sk, se, m)
            g["rel_k"] = _rel_sens(sk, y, m)
            g["rel_e"] = _rel_sens(se, y, m)
            g["n"] = int(m.sum())
            summary[name][obs] = g

    print("\nderived sub-observables (final step)", flush=True)
    for name in points:
        print(f"  point '{name}'", flush=True)
        print(f"    {'observable':>12s} {'cells':>7s} {'rel_k':>10s} {'rel_e':>10s} "
              f"{'r':>8s} {'cond':>11s}", flush=True)
        for obs in DERIVED:
            g = summary[name][obs]
            print(f"    {obs:>12s} {g['n']:7d} {g['rel_k']:10.3e} {g['rel_e']:10.3e} "
                  f"{g['r']:8.4f} {g['cond']:11.4e}", flush=True)

    combo_stats = compute_combo_stats(d, masks)
    np.save(os.path.join(out_dir, "combo_stats.npy"),
            np.array([combo_stats], dtype=object), allow_pickle=True)
    print("\nvariance-normalised joint-loss conditioning (final step)", flush=True)
    for name, rows in combo_stats.items():
        print(f"  point '{name}'", flush=True)
        print(f"    {'combination':>44s} {'lambda_min':>12s} {'cond':>12s} {'r':>9s}", flush=True)
        for combo, st in rows.items():
            print(f"    {' + '.join(combo):>44s} {st['lmin']:12.4e} {st['cond']:12.4e} {st['r']:9.4f}",
                  flush=True)

    # --- Figure 1: surface log-sensitivity maps ------------------------------
    # A handful of cells carry sensitivities orders of magnitude above the rest
    # (see the concentration figure), so a linear full-range colour scale renders
    # a blank map. Limits are clipped at the 98th percentile of |S| per column.
    map_fields = [
        ("temp", "surface temp (deg C)"),
        ("salt", "surface salt (g/kg)"),
        ("Nsqr", "surface Nsqr (1/s^2)"),
        ("mld", "MLD (m)"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(22, 8.5), sharex=True, sharey=True)
    for j, (field, label) in enumerate(map_fields):
        if field == "mld":
            m2 = S["mld_mask"]
            sk, se = masked(S["S_k"][field], m2), masked(S["S_e"][field], m2)
        else:
            m2 = MASK_NP[:, :, SURFACE]
            sk = masked(S["S_k"][field][:, :, SURFACE], m2)
            se = masked(S["S_e"][field][:, :, SURFACE], m2)
        v = np.nanpercentile(np.abs(np.concatenate([sk[~np.isnan(sk)], se[~np.isnan(se)]])), 98)
        v = v if v > 0 else 1.0
        for i, (s, pname) in enumerate(((sk, "c_k"), (se, "c_eps"))):
            im = axs[i, j].pcolormesh(XT, YT, s.T, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
            axs[i, j].set_title(f"d({label}) / d(log {pname})", fontsize=10)
            if i == 1:
                axs[i, j].set_xlabel("longitude")
        fig.colorbar(im, ax=[axs[0, j], axs[1, j]], shrink=0.75, pad=0.02, extend="both")
    axs[0, 0].set_ylabel("latitude")
    axs[1, 0].set_ylabel("latitude")
    fig.suptitle(f"Report 17: log-sensitivity of surface observables after {N_STEPS} steps "
                 f"(at c_k={ck_ref}, c_eps={ce_ref}); shared scale per column, clipped at the 98th percentile")
    p = os.path.join(out_dir, "surface_sensitivity_maps.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 2: zonal-mean depth sections, log colour scale ---------------
    sect_fields = [("temp", "temp"), ("salt", "salt"), ("Nsqr", "Nsqr"), ("tke", "tke"), ("dTdz", "dT/dz")]
    fig, axs = plt.subplots(2, len(sect_fields), figsize=(5 * len(sect_fields), 8), sharex=True, sharey=True)
    for j, (field, label) in enumerate(sect_fields):
        m = field_mask(field, S["mld_mask"])
        zax = 0.5 * (ZT_NP[1:] + ZT_NP[:-1]) if FIELD_KIND[field] == "W" else ZT_NP
        secs = []
        for key in ("S_k", "S_e"):
            with np.errstate(invalid="ignore"):
                s = np.where(m, np.abs(S[key][field]), np.nan)
            with np.errstate(invalid="ignore"):
                secs.append(np.nanmean(s, axis=0))
        vmax = np.nanmax(np.abs(np.concatenate([np.ravel(x) for x in secs])))
        norm = LogNorm(vmin=max(vmax * 1e-5, 1e-300), vmax=max(vmax, 1e-299))
        for i, (sec, pname) in enumerate(zip(secs, ("c_k", "c_eps"))):
            im = axs[i, j].pcolormesh(YT, zax, np.where(sec > 0, sec, np.nan).T,
                                      cmap="magma", norm=norm, shading="auto")
            axs[i, j].set_title(f"|d({label}) / d(log {pname})|", fontsize=10)
            if i == 1:
                axs[i, j].set_xlabel("latitude")
        fig.colorbar(im, ax=[axs[0, j], axs[1, j]], shrink=0.75, pad=0.02)
    axs[0, 0].set_ylabel("depth (m)")
    axs[1, 0].set_ylabel("depth (m)")
    axs[0, 0].set_ylim(-2600, 0)  # the c_k signal is still ~10% of its surface value at 2500 m
    fig.suptitle(f"Report 17: zonal-mean |log-sensitivity| by depth after {N_STEPS} steps "
                 f"(at c_k={ck_ref}, c_eps={ce_ref}); log colour scale, shared per column, top 2600 m")
    p = os.path.join(out_dir, "depth_sections.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 3: sensitivity magnitude and separability by observable ------
    fig, axs = plt.subplots(1, 3, figsize=(24, 6.5))
    x = np.arange(len(ALL_OBS))
    axs[0].bar(x - 0.2, [summary[REF][f]["rel_k"] for f in ALL_OBS], 0.4, label="c_k", color="tab:blue")
    axs[0].bar(x + 0.2, [summary[REF][f]["rel_e"] for f in ALL_OBS], 0.4, label="c_eps", color="tab:red")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("rel. sensitivity  rms(dy/dlog theta) / std(y)")
    axs[0].set_title("Sensitivity magnitude (dimensionless)")
    axs[0].legend()

    axs[1].bar(x, [abs(summary[REF][f]["r"]) for f in ALL_OBS], color="tab:purple")
    axs[1].axhline(1.0, color="k", lw=0.8)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("|r| = |<S_k,S_e>| / (|S_k||S_e|)")
    axs[1].set_title("Collinearity of the two parameter directions\n(1 = indistinguishable)")

    axs[2].bar(x, [summary[REF][f]["cond"] for f in ALL_OBS], color="tab:green")
    axs[2].set_yscale("log")
    axs[2].set_ylabel("cond(H) = lambda_max / lambda_min")
    axs[2].set_title("Conditioning of the 2x2 Gauss-Newton matrix\n(1 = perfectly conditioned)")

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_OBS, rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Report 17: which observable can identify c_k and c_eps after {N_STEPS} steps "
                 f"(at c_k={ck_ref}, c_eps={ce_ref})")
    fig.tight_layout()
    p = os.path.join(out_dir, "identifiability_by_observable.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 4: signal onset over the rollout -----------------------------
    steps = np.arange(1, N_STEPS + 1)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    for field in TS_FIELDS:
        ts = timeseries[REF][field]
        axs[0].plot(steps, ts["rel_k"], "-o", ms=3, label=field)
        axs[1].plot(steps, ts["rel_e"], "-o", ms=3, label=field)
        axs[2].plot(steps, np.abs(ts["r"]), "-o", ms=3, label=field)
    axs[0].set_title("relative sensitivity to c_k")
    axs[1].set_title("relative sensitivity to c_eps\n(exactly zero for every field at step 1;\nfor everything but tke also at step 2)")
    axs[2].set_title("|collinearity r|  (1 = the two params are indistinguishable)")
    axs[2].axhline(1.0, color="k", lw=0.8)
    axs[2].set_ylim(0, 1.05)
    for ax in axs[:2]:
        ax.set_yscale("log")
        ax.set_ylabel("rms(dy/dlog theta) / std(y)")
    for ax in axs:
        ax.set_xlabel("rollout step (days)")
        ax.set_xticks(np.arange(0, N_STEPS + 1, 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Report 17: onset of parameter signal over the rollout (at c_k={ck_ref}, c_eps={ce_ref})")
    fig.tight_layout()
    p = os.path.join(out_dir, "signal_onset.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 5: vertical breakdown ---------------------------------------
    lvl_fields = d["lvl_fields"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    im0 = axs[0].pcolormesh(np.arange(NZ), np.arange(len(lvl_fields)), np.log10(d["rel_e_lvl"]),
                            cmap="viridis", shading="auto")
    axs[0].set_title("log10 relative sensitivity to c_eps, by level")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)
    im1 = axs[1].pcolormesh(np.arange(NZ), np.arange(len(lvl_fields)), d["r_lvl"],
                            cmap="magma_r", vmin=0, vmax=1, shading="auto")
    axs[1].set_title("|collinearity r|, by level")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)
    for ax in axs:
        ax.set_yticks(np.arange(len(lvl_fields)))
        ax.set_yticklabels(lvl_fields)
        ax.set_xticks(np.arange(NZ))
        ax.set_xticklabels([f"{z:.0f}" for z in ZT_NP], rotation=90, fontsize=7)
        ax.set_xlabel("level depth (m), surface at right")
    fig.suptitle(f"Report 17: vertical breakdown of the c_eps signal after {N_STEPS} steps "
                 f"(at c_k={ck_ref}, c_eps={ce_ref})")
    fig.tight_layout()
    p = os.path.join(out_dir, "levelwise_breakdown.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 6: joint-loss conditioning and the weak direction ------------
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[1.1, 1])
    combos = list(combo_stats[REF].keys())
    labels = [" + ".join(c) for c in combos]
    conds = [combo_stats[REF][c]["cond"] for c in combos]
    lmins = [combo_stats[REF][c]["lmin"] for c in combos]
    xs = np.arange(len(combos))
    axs[0].barh(xs, lmins, color="tab:green")
    axs[0].set_xscale("log")
    axs[0].set_xlim(right=max(lmins) * 60)
    axs[0].set_yticks(xs)
    axs[0].set_yticklabels(labels, fontsize=9)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("lambda_min(H): curvature along the worst-determined parameter direction")
    axs[0].set_title("Joint-loss conditioning, variance-normalised\n"
                     "(bar = lambda_min, bigger is better; it spans 4 orders of magnitude)")
    axs[0].grid(alpha=0.3, axis="x")
    for xi, (cnd, lm) in enumerate(zip(conds, lmins)):
        axs[0].text(lm * 1.6, xi, f"cond={cnd:.1f}", va="center", fontsize=8)

    # candidates restricted to observables with one value per column, so the
    # per-cell score can be drawn as a map
    best_field = max(("temp_surf", "mld", "Nsqr_surf", "prho_avg3", "dTdz_absavg3", "dTdz_avg3"),
                     key=lambda f: summary[REF][f]["lmin"])
    v = summary[REF][best_field]["v_min"]
    m2, sk, se, _ = masks(d, S, best_field)
    score = np.where(m2, (v[0] * sk + v[1] * se) ** 2, np.nan)
    im = axs[1].pcolormesh(XT, YT, score.T, cmap="inferno", shading="auto",
                           vmin=0, vmax=np.nanpercentile(score, 99))
    axs[1].set_title(f"per-cell information about the weak direction\n"
                     f"({best_field}; weak eigenvector = [{v[0]:.2f}, {v[1]:.2f}] in (log c_k, log c_eps))")
    axs[1].set_xlabel("longitude")
    axs[1].set_ylabel("latitude")
    fig.colorbar(im, ax=axs[1], shrink=0.85, extend="max")
    fig.suptitle(f"Report 17: designing a loss that identifies both parameters (at c_k={ck_ref}, c_eps={ce_ref})")
    fig.tight_layout()
    p = os.path.join(out_dir, "joint_loss_design.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"saved {p}", flush=True)

    # --- Figure 7: how local is the local analysis? --------------------------
    if len(points) > 1:
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        x = np.arange(len(ALL_OBS))
        width = 0.8 / len(points)
        for i, name in enumerate(points):
            off = (i - (len(points) - 1) / 2) * width
            lbl = f"{name} ({d['point_params'][name][0]}, {d['point_params'][name][1]})"
            axs[0].bar(x + off, [summary[name][f]["rel_e"] for f in ALL_OBS], width, label=lbl)
            axs[1].bar(x + off, [summary[name][f]["cond"] for f in ALL_OBS], width, label=lbl)
        axs[0].set_yscale("log")
        axs[0].set_ylabel("rel. sensitivity to c_eps")
        axs[0].set_title("c_eps signal strength across evaluation points")
        axs[1].set_yscale("log")
        axs[1].set_ylabel("cond(H)")
        axs[1].set_title("Conditioning across evaluation points")
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels(ALL_OBS, rotation=45, ha="right")
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle("Report 17: local tangents are local -- the same analysis at the truth, "
                     "at reports 15/16's start, and at report-15's c_eps stall point")
        fig.tight_layout()
        p = os.path.join(out_dir, "across_points.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"saved {p}", flush=True)

    # --- Figure 9: the 2D stratification metric ------------------------------
    # mean(|dT/dz|) over the top 3 interfaces: one number per column, so it maps
    # like MLD does, but smooth -- no discrete level selection anywhere in it.
    strat = [
        ("dTdz_absavg3", "mean |dT/dz|, top 3  (deg C / m)"),
        ("dSdz_absavg3", "mean |dS/dz|, top 3  ((g/kg) / m)"),
        ("drhodz_absavg3", "mean |d(rho)/dz|, top 3  ((kg/m^3) / m)"),
        ("mld", "MLD (m)  -- for reference"),
    ]
    fig, axs = plt.subplots(3, len(strat), figsize=(5.5 * len(strat), 12), sharex=True, sharey=True)
    for j, (obs, label) in enumerate(strat):
        m2, sk, se, y = masks(d, S, obs)
        prim = np.where(m2, y, np.nan)
        im = axs[0, j].pcolormesh(XT, YT, prim.T, cmap="viridis", shading="auto",
                                  vmin=np.nanpercentile(prim, 2), vmax=np.nanpercentile(prim, 98))
        axs[0, j].set_title(label, fontsize=10)
        fig.colorbar(im, ax=axs[0, j], shrink=0.85, extend="both")
        for i, (sig, pname) in enumerate(((sk, "c_k"), (se, "c_eps")), start=1):
            f = np.where(m2, sig, np.nan)
            v = np.nanpercentile(np.abs(f[~np.isnan(f)]), 98) or 1.0
            im = axs[i, j].pcolormesh(XT, YT, f.T, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
            g = summary[REF][obs]
            axs[i, j].set_title(f"d / d(log {pname})   (rel. sens. {g['rel_k' if i == 1 else 'rel_e']:.2e})",
                                fontsize=9)
            fig.colorbar(im, ax=axs[i, j], shrink=0.85, extend="both")
        axs[2, j].set_xlabel("longitude")
    for i, lab in enumerate(("value", "sensitivity to c_k", "sensitivity to c_eps")):
        axs[i, 0].set_ylabel(f"{lab}\nlatitude")
    fig.suptitle(f"Report 17: column stratification metrics after {N_STEPS} steps "
                 f"(at c_k={ck_ref}, c_eps={ce_ref}); sensitivity scales clipped at the 98th percentile")
    p_ = os.path.join(out_dir, "stratification_metric.png")
    fig.savefig(p_, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_}", flush=True)

    # --- Figure 8: how concentrated is the signal? ---------------------------
    # A loss whose gradient comes from a handful of cells is the failure mode
    # report-16 diagnosed for MLD. This measures it directly, on the tangents.
    conc_fields = ["temp_surf", "salt", "mld", "dTdz_top3", "dTdz_absavg3", "prho_avg3", "tke"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, key, pname in ((axs[0], "S_k", "c_k"), (axs[1], "S_e", "c_eps")):
        for field in conc_fields:
            m2, sk_, se_, _ = masks(d, S, field)
            w = np.sort(((sk_ if key == "S_k" else se_)[m2]) ** 2)[::-1]
            tot = w.sum()
            if tot <= 0:
                continue
            frac = np.cumsum(w) / tot
            ax.plot(100.0 * np.arange(1, w.size + 1) / w.size, 100.0 * frac, label=field)
        ax.set_xscale("log")
        ax.set_xlabel("% of cells, ranked by squared sensitivity")
        ax.set_ylabel(f"% of total squared sensitivity to {pname}")
        ax.set_title(f"Concentration of the {pname} signal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.axhline(90, color="k", lw=0.8, ls="--")
    fig.suptitle(f"Report 17: how much of each observable's parameter signal lives in how few cells "
                 f"(after {N_STEPS} steps, at c_k={ck_ref}, c_eps={ce_ref}); dashed line = 90%")
    fig.tight_layout()
    p = os.path.join(out_dir, "signal_concentration.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(os.path.dirname(here)), "figures", "report-17")
    make_figures(sys.argv[1] if len(sys.argv) > 1 else default)
