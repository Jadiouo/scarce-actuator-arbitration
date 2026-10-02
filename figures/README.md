# figures/ index

Replot everything from existing JSON only (CPU, no experiments): `make replot`.

Status: **main** = supports a claim in docs/claims.md; **exploratory**; **superseded** = historical, does not represent final conclusions (see docs/claims.md C1).
Figures 1-8 and 11-13 come from `arbitration/figures.py`; 9b/10b from `scripts/run_dp_step.py figs`; 14a/14b from `scripts/run_c_pricing.py figs`; 15a/15b from `scripts/run_d_scaling.py figs` (scripts now write the final filenames). Fig1-3 use legacy-model numbers (e.g. fig3 "+48% vs honest") and do not represent the final conclusions.

| File | Content | Data (JSON key) | Cell | Status |
|---|---|---|---|---|
| fig1_precision_stops_predicting.png | legacy: precision of prediction vs health | results.json load_sweep | N=8 legacy | historical / superseded |
| fig2_information_has_no_value.png | legacy: policies vs round-robin ("information worthless") | results.json load_sweep | N=8 legacy | historical / superseded (C1: conclusion driven by ignoring distance) |
| fig3_pricing_the_request.png | legacy pricing gain | results.json | N=8 legacy | historical / superseded (see fig14a, C7) |
| fig4_sigma_sweep.png | noise sigma sweep | results.json | N=8 legacy | historical, exploratory |
| fig5_lambda_bracketing.png | lambda bracketing | results.json | N=8 legacy | historical, exploratory |
| fig6_hysteresis.png | hysteresis | results.json | N=8 legacy | historical, exploratory |
| fig7_v2_load_sweep.png | v2 load sweep, policies vs r | results.json v2_sweep | N=8 v2 | main (C1) |
| fig8_legacy_vs_v2_gap.png | gap to best policy, legacy vs v2 | results.json v2_sweep | N=8 | main (C1) |
| fig9_optimality_gap.png | round-robin gap to DP | results.json dp_sweep | N<=4 | older DP; superseded by fig9b |
| fig9b_gap_distribution.png | gap distribution vs dp_step_full | dp_step.json | N=3,4 | main (C3) |
| fig10_voi_regions.png | VoI regions (older DP) | results.json dp_sweep | N=3 | superseded by fig10b |
| fig10b_voi_step_regions.png | VoI_step vs shock_frac / spread | dp_step.json | N=3, r=.004, K=(20,30), 16 seeds | main (C2) |
| fig11_strategic_loss_v2.png | truthful vs approx. best-response profile health, 14 mechanisms, 95% CI | game.json game_v2.cells[N3_r0.004_f0.5, N8_r0.002_f0.5].mech.*.games["full\|truth\|eps=0.002"] | N=3 r=.004 f=.5 57 actions; N=8 r=.002 f=.5 24 actions; 32 seeds | main (C4-C6) |
| fig12_equilibrium_b_v2.png | final mean inflation b and convergence rate | same | same | main (convergence caveat of C5/C6) |
| fig13_strategy_families.png | loss per strategy family heatmap | game.json game_v2.cells.*.mech.*.games | same two cells | main (C5/C6), loss = truthful - eq, * = CI excludes 0 |
| fig14a_priced_vs_nothrash.png | priced gain vs no-thrash baseline | c_pricing.json | N=8, lambda* from seeds 0-15, holdout | exploratory (C7) |
| fig14b_implied_exchange_rate.png | implied lambda / exchange rate | c_pricing.json | N=8 | exploratory (C7) |
| fig15a_scaling_with_N.png | scaling with N; (c) loss, (d) 1 vs 2 actuators | d_scaling.json | load calibrated per N, 25 actions | exploratory (C4, C8) |
| fig15b_voi_proxy_load_schemes.png | VoI proxy under load schemes | d_scaling.json | N=3..16 | exploratory (C8) |
