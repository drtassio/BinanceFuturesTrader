"""
FASE 6 — Testes Científicos & Matemáticos

Bateria de testes formais validados contra a literatura, executados sobre
DUAS estratégias sintéticas:
  - skill: drift +0.0008, vol 0.01 (Sharpe esperado >> 0)
  - noise: drift 0.0, vol 0.01 (Sharpe esperado ≈ 0)

Para cada teste, o critério é "skill PASSA, noise FALHA" — assim validamos
que o teste DISTINGUE corretamente.

Testes:
  6.1  Sharpe Ratio direto
  6.2  Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
  6.3  White's Reality Check (bootstrap correto)
  6.4  Block-bootstrap permutation test
  6.5  ADF / KPSS para estacionariedade
  6.6  PBO (Probability of Backtest Overfitting) — combinatorial reduzido
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from scripts.validation._common import PhaseRunner


# ---------------------------------------------------------------------------
# Implementações matemáticas
# ---------------------------------------------------------------------------
def deflated_sharpe_ratio(sr_observed: float, n_trials: int, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0) -> dict:
    """
    DSR closed-form (Bailey & López de Prado, 2014, eq. 9):
        DSR = Φ((SR_obs − E[max SR_N]) · sqrt(T−1) / sqrt(1 − γ₃·SR + (γ₄−1)/4·SR²))

    onde E[max SR_N] = σ(SR) · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))).
    SR aqui é NÃO-anualizado (por período).
    """
    if n_trials < 2 or n_obs < 30:
        return {"dsr": None, "note": "insufficient data"}
    emc = 0.5772156649  # Euler-Mascheroni
    # σ(SR) sob H0 (SR=0); para SR>0 o termo de correção é pequeno
    var_sr_h0 = 1.0 / (n_obs - 1)
    sigma_sr_h0 = float(np.sqrt(var_sr_h0))
    # E[max SR] entre N trials independentes ~ N(0, σ_SR_H0²)
    multiplier = ((1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
                  + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr_expected_max = sigma_sr_h0 * multiplier
    # σ(SR) sob H1 (observado): correção para skew/kurt da distribuição empírica
    var_sr_obs = (1 - skew * sr_observed + ((kurt - 1) / 4) * sr_observed ** 2) / (n_obs - 1)
    sigma_sr_obs = float(np.sqrt(max(var_sr_obs, 1e-12)))
    z = (sr_observed - sr_expected_max) / sigma_sr_obs
    dsr = float(stats.norm.cdf(z))
    return {"dsr": dsr, "z": float(z), "sr_expected_max": float(sr_expected_max),
            "sigma_sr_obs": sigma_sr_obs, "sr_observed": float(sr_observed)}


def whites_reality_check(diff: np.ndarray, n_bootstrap: int = 1000,
                         block_size: int = 20, seed: int = 42) -> dict:
    """
    White (2000) — bootstrap estacionário por blocos (Politis & Romano, 1994).
    H0: mean(diff) <= 0.  Retorna p-value one-sided.
    """
    rng = np.random.default_rng(seed)
    n = len(diff)
    obs_mean = float(diff.mean())
    # Centra para simular H0 (White, 2000, Sec. 3)
    centered = diff - obs_mean

    # Bootstrap estacionário com blocos contíguos
    n_blocks_needed = int(np.ceil(n / block_size))
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks_needed)
        boot = np.concatenate([
            centered[s:s + block_size] if s + block_size <= n
            else np.concatenate([centered[s:], centered[:s + block_size - n]])
            for s in starts
        ])[:n]
        boot_means[i] = boot.mean()
    p_value = float((boot_means >= obs_mean).mean())
    return {"observed_mean": obs_mean, "p_value": p_value, "n_bootstrap": n_bootstrap}


def block_permutation_sharpe(returns: np.ndarray, block_size: int = 20,
                             n_perms: int = 500, periods_per_year: float = 252 * 24,
                             seed: int = 42) -> dict:
    """
    Permutação por blocos: testa se Sharpe observado > Sharpe sob permutação aleatória.
    Centra ANTES de permutar para simular H0 (mean=0).
    """
    rng = np.random.default_rng(seed)
    sharpe = lambda r: float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year))
    sr_real = sharpe(returns)
    centered = returns - returns.mean()  # sob H0, drift = 0
    n = len(centered)
    n_blocks = n // block_size
    perms = np.empty(n_perms)
    for i in range(n_perms):
        blocks = [centered[j * block_size:(j + 1) * block_size] for j in range(n_blocks)]
        rng.shuffle(blocks)
        perms[i] = sharpe(np.concatenate(blocks))
    p_value = float((perms >= sr_real).mean())
    return {"sharpe_observed": sr_real, "perm_mean": float(perms.mean()),
            "perm_std": float(perms.std()), "p_value": p_value}


def pbo_combinatorial(returns_matrix: np.ndarray, n_splits: int = 10) -> dict:
    """
    PBO (Bailey, Borwein, López de Prado, 2014) — combinatorial.
    returns_matrix: (T, N_strategies). PBO ≈ 0.5 para estratégias sem skill;
    PBO baixo (<0.5) indica que o vencedor IS continua sendo bom OOS.
    """
    T, N = returns_matrix.shape
    if T < n_splits * 4 or N < 2:
        return {"pbo": None, "note": "insufficient data"}
    split_size = T // n_splits
    splits = [returns_matrix[i * split_size:(i + 1) * split_size] for i in range(n_splits)]
    from itertools import combinations
    logits = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_data = np.concatenate([splits[i] for i in is_idx], axis=0)
        oos_data = np.concatenate([splits[i] for i in oos_idx], axis=0)
        sr_is = is_data.mean(axis=0) / (is_data.std(axis=0) + 1e-12)
        sr_oos = oos_data.mean(axis=0) / (oos_data.std(axis=0) + 1e-12)
        n_star = int(np.argmax(sr_is))
        # ranking de OOS: fração de outras estratégias com SR_oos > SR_oos[n_star]
        rank_oos = (sr_oos > sr_oos[n_star]).sum() / max(N - 1, 1)
        rank_oos = float(np.clip(rank_oos, 1e-3, 1 - 1e-3))
        logits.append(np.log(rank_oos / (1 - rank_oos)))
    pbo = float((np.array(logits) > 0).mean())
    return {"pbo": pbo, "n_partitions": len(logits)}


# ---------------------------------------------------------------------------
# Fase
# ---------------------------------------------------------------------------
def run() -> PhaseRunner:
    p = PhaseRunner("6 — Testes Científicos")
    with p:
        rng = np.random.default_rng(42)
        n = 5000
        periods_per_year = 252 * 24  # horário

        ret_skill = rng.normal(0.0008, 0.01, n)
        ret_noise = rng.normal(0.0000, 0.01, n)
        ret_bench = rng.normal(0.0000, 0.01, n)

        sr_skill_per = float(ret_skill.mean() / (ret_skill.std() + 1e-12))
        sr_noise_per = float(ret_noise.mean() / (ret_noise.std() + 1e-12))
        sr_skill_ann = sr_skill_per * np.sqrt(periods_per_year)
        sr_noise_ann = sr_noise_per * np.sqrt(periods_per_year)

        p.metric("sharpe", {"skill_annualized": sr_skill_ann, "noise_annualized": sr_noise_ann})

        # 6.1 — Sharpe direto: skill > 1, noise < 1
        p.check(
            "Sharpe: skill anualizado > 1.0",
            sr_skill_ann > 1.0, detail=f"SR_skill={sr_skill_ann:.2f}",
        )
        p.check(
            "Sharpe: noise anualizado próximo de 0 (|SR| < 1.0)",
            abs(sr_noise_ann) < 1.0, detail=f"SR_noise={sr_noise_ann:.2f}",
        )

        # 6.2 — DSR (Bailey & López de Prado): usa SR POR PERÍODO
        dsr_skill = deflated_sharpe_ratio(sr_skill_per, n_trials=100, n_obs=n)
        dsr_noise = deflated_sharpe_ratio(sr_noise_per, n_trials=100, n_obs=n)
        p.metric("dsr_skill", dsr_skill)
        p.metric("dsr_noise", dsr_noise)
        p.check(
            "DSR distingue skill (dsr_skill > dsr_noise)",
            dsr_skill["dsr"] > dsr_noise["dsr"],
            detail=f"skill={dsr_skill['dsr']:.3f} vs noise={dsr_noise['dsr']:.3f}",
        )

        # 6.3 — White's Reality Check
        wrc_skill = whites_reality_check(ret_skill - ret_bench, n_bootstrap=500, block_size=20)
        wrc_noise = whites_reality_check(ret_noise - ret_bench, n_bootstrap=500, block_size=20)
        p.metric("wrc", {"skill": wrc_skill, "noise": wrc_noise})
        p.check(
            "WRC: skill p < 0.05 (rejeita H0 'sem edge')",
            wrc_skill["p_value"] < 0.05,
            detail=f"p_skill={wrc_skill['p_value']:.4f}",
        )
        p.check(
            "WRC: noise p >= 0.05 (não rejeita H0)",
            wrc_noise["p_value"] >= 0.05,
            detail=f"p_noise={wrc_noise['p_value']:.4f}",
        )

        # 6.4 — Block-bootstrap permutation test
        bpt_skill = block_permutation_sharpe(ret_skill, block_size=20, n_perms=500,
                                             periods_per_year=periods_per_year)
        bpt_noise = block_permutation_sharpe(ret_noise, block_size=20, n_perms=500,
                                             periods_per_year=periods_per_year)
        p.metric("block_permutation", {"skill": bpt_skill, "noise": bpt_noise})
        p.check(
            "Block-perm: skill p < 0.05",
            bpt_skill["p_value"] < 0.05,
            detail=f"p_skill={bpt_skill['p_value']:.4f}",
        )
        p.check(
            "Block-perm: noise p >= 0.05",
            bpt_noise["p_value"] >= 0.05,
            detail=f"p_noise={bpt_noise['p_value']:.4f}",
        )

        # 6.5 — Estacionariedade (ADF)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf = adfuller(ret_skill, autolag="AIC")
            p.metric("adf_skill", {"stat": float(adf[0]), "pvalue": float(adf[1])})
            p.check(
                "ADF: retornos estacionários (p < 0.05)",
                adf[1] < 0.05, detail=f"stat={adf[0]:.3f} p={adf[1]:.4f}",
            )
        except Exception as e:
            p.check("ADF disponível", False, detail=str(e))

        # 6.6 — PBO
        # Estratégias aleatórias: PBO esperado ≈ 0.5 (skill nula → vencedor IS é aleatório OOS)
        N = 50
        rand_strats = rng.normal(0.0, 0.01, (n, N))
        pbo_rand = pbo_combinatorial(rand_strats, n_splits=10)
        p.metric("pbo_random", pbo_rand)
        p.check(
            "PBO em estratégias aleatórias está em [0.3, 0.7] (próximo de 0.5)",
            pbo_rand.get("pbo") is not None and 0.30 <= pbo_rand["pbo"] <= 0.70,
            detail=f"pbo={pbo_rand.get('pbo'):.3f}",
        )

        # Estratégias com skill heterogêneo: PBO deve ser baixo (<0.5)
        N = 50
        # Cada coluna tem mu próprio uniformemente espaçado entre 0 e 0.0015
        mus = np.linspace(0, 0.0015, N)
        skill_strats = rng.normal(mus, 0.01, (n, N))
        pbo_skill = pbo_combinatorial(skill_strats, n_splits=10)
        p.metric("pbo_skill", pbo_skill)
        p.check(
            "PBO em estratégias com skill heterogêneo < 0.5",
            pbo_skill.get("pbo") is not None and pbo_skill["pbo"] < 0.5,
            detail=f"pbo={pbo_skill.get('pbo'):.3f}",
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
