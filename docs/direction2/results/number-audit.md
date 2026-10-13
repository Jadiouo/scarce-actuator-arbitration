# 方向二主實驗數字稽核

共 112 項 CI／MDE 從原始 snapshot 重算，全部一致。判定另以獨立條件式核對為 J3。

原始檔：results/direction2/main.json；以下 JSON 路徑均相對其根節點。逐 seed 原始資料位於 results/direction2/main_eval/。

| JSON 路徑 | 數字 | 重算一致 |
|---|---|---|
| `cells[0].runs[0].G_RL` | +0.03452 [+0.02997, +0.03908] | 是 |
| `cells[0].runs[0].D` | +0.01366 [+0.00932, +0.01800] | 是 |
| `cells[0].runs[0].D_plus_h0` | +0.01366 [+0.00932, +0.01800] | 是 |
| `cells[0].runs[0].system.regret_per_round` | +0.01957 [+0.01794, +0.02121] | 是 |
| `cells[0].runs[0].system.mean_ineligible_fraction` | +0.06467 [+0.05690, +0.07244] | 是 |
| `cells[0].runs[0].system.ineligible_fraction_by_agent[0]` | +0.02393 [+0.01456, +0.03331] | 是 |
| `cells[0].runs[0].system.ineligible_fraction_by_agent[1]` | +0.13306 [+0.11475, +0.15138] | 是 |
| `cells[0].runs[0].system.ineligible_fraction_by_agent[2]` | +0.03702 [+0.02619, +0.04784] | 是 |
| `cells[0].runs[0].MDE_G` | 0.00608054 | 是 |
| `cells[0].runs[0].MDE_D` | 0.00481471 | 是 |
| `cells[0].runs[0].MDE_D_plus_h0` | 0.00481471 | 是 |
| `cells[0].runs[1].G_RL` | +0.03399 [+0.02984, +0.03814] | 是 |
| `cells[0].runs[1].D` | +0.01313 [+0.00925, +0.01700] | 是 |
| `cells[0].runs[1].D_plus_h0` | +0.01313 [+0.00925, +0.01700] | 是 |
| `cells[0].runs[1].system.regret_per_round` | +0.01942 [+0.01746, +0.02139] | 是 |
| `cells[0].runs[1].system.mean_ineligible_fraction` | +0.06642 [+0.05781, +0.07502] | 是 |
| `cells[0].runs[1].system.ineligible_fraction_by_agent[0]` | +0.02714 [+0.01668, +0.03759] | 是 |
| `cells[0].runs[1].system.ineligible_fraction_by_agent[1]` | +0.13292 [+0.11682, +0.14902] | 是 |
| `cells[0].runs[1].system.ineligible_fraction_by_agent[2]` | +0.03919 [+0.02701, +0.05137] | 是 |
| `cells[0].runs[1].MDE_G` | 0.00611793 | 是 |
| `cells[0].runs[1].MDE_D` | 0.00613413 | 是 |
| `cells[0].runs[1].MDE_D_plus_h0` | 0.00613413 | 是 |
| `cells[0].runs[2].G_RL` | +0.03537 [+0.03139, +0.03934] | 是 |
| `cells[0].runs[2].D` | +0.01450 [+0.01034, +0.01865] | 是 |
| `cells[0].runs[2].D_plus_h0` | +0.01450 [+0.01034, +0.01865] | 是 |
| `cells[0].runs[2].system.regret_per_round` | +0.01988 [+0.01807, +0.02169] | 是 |
| `cells[0].runs[2].system.mean_ineligible_fraction` | +0.06531 [+0.05754, +0.07309] | 是 |
| `cells[0].runs[2].system.ineligible_fraction_by_agent[0]` | +0.02634 [+0.01600, +0.03667] | 是 |
| `cells[0].runs[2].system.ineligible_fraction_by_agent[1]` | +0.13180 [+0.11568, +0.14791] | 是 |
| `cells[0].runs[2].system.ineligible_fraction_by_agent[2]` | +0.03781 [+0.02648, +0.04913] | 是 |
| `cells[0].runs[2].MDE_G` | 0.00592153 | 是 |
| `cells[0].runs[2].MDE_D` | 0.00612992 | 是 |
| `cells[0].runs[2].MDE_D_plus_h0` | 0.00612992 | 是 |
| `cells[0].honest_system.regret_per_round` | +0.00686 [+0.00506, +0.00866] | 是 |
| `cells[0].honest_system.mean_ineligible_fraction` | +0.02776 [+0.02143, +0.03408] | 是 |
| `cells[0].honest_system.ineligible_fraction_by_agent[0]` | +0.02324 [+0.01345, +0.03302] | 是 |
| `cells[0].honest_system.ineligible_fraction_by_agent[1]` | +0.02201 [+0.01394, +0.03008] | 是 |
| `cells[0].honest_system.ineligible_fraction_by_agent[2]` | +0.03802 [+0.02555, +0.05049] | 是 |
| `cells[0].G_HW.public` | +0.02087 [+0.01679, +0.02495] | 是 |
| `cells[0].G_HW.plus_h0` | +0.02087 [+0.01679, +0.02495] | 是 |
| `cells[0].G_HW.including_oracle` | +0.08646 [+0.08097, +0.09195] | 是 |
| `cells[0].hw_system[honest].regret_per_round` | +0.00686 [+0.00506, +0.00866] | 是 |
| `cells[0].hw_system[honest].mean_ineligible_fraction` | +0.02776 [+0.02143, +0.03408] | 是 |
| `cells[0].hw_system[honest].ineligible_fraction_by_agent[0]` | +0.02324 [+0.01345, +0.03302] | 是 |
| `cells[0].hw_system[honest].ineligible_fraction_by_agent[1]` | +0.02201 [+0.01394, +0.03008] | 是 |
| `cells[0].hw_system[honest].ineligible_fraction_by_agent[2]` | +0.03802 [+0.02555, +0.05049] | 是 |
| `cells[0].hw_system[dz=0.3].regret_per_round` | +0.01534 [+0.01325, +0.01743] | 是 |
| `cells[0].hw_system[dz=0.3].mean_ineligible_fraction` | +0.06850 [+0.05999, +0.07701] | 是 |
| `cells[0].hw_system[dz=0.3].ineligible_fraction_by_agent[0]` | +0.02418 [+0.01465, +0.03371] | 是 |
| `cells[0].hw_system[dz=0.3].ineligible_fraction_by_agent[1]` | +0.14832 [+0.13153, +0.16512] | 是 |
| `cells[0].hw_system[dz=0.3].ineligible_fraction_by_agent[2]` | +0.03300 [+0.02002, +0.04598] | 是 |
| `cells[0].hw_system[edge_oracle em=0.2].regret_per_round` | +0.02882 [+0.02708, +0.03056] | 是 |
| `cells[0].hw_system[edge_oracle em=0.2].mean_ineligible_fraction` | +0.06937 [+0.06192, +0.07683] | 是 |
| `cells[0].hw_system[edge_oracle em=0.2].ineligible_fraction_by_agent[0]` | +0.02313 [+0.01397, +0.03229] | 是 |
| `cells[0].hw_system[edge_oracle em=0.2].ineligible_fraction_by_agent[1]` | +0.15383 [+0.13713, +0.17052] | 是 |
| `cells[0].hw_system[edge_oracle em=0.2].ineligible_fraction_by_agent[2]` | +0.03117 [+0.01969, +0.04264] | 是 |
| `cells[1].runs[0].G_RL` | +0.01296 [+0.01095, +0.01497] | 是 |
| `cells[1].runs[0].D` | +0.01163 [+0.00911, +0.01415] | 是 |
| `cells[1].runs[0].D_plus_h0` | +0.01163 [+0.00911, +0.01415] | 是 |
| `cells[1].runs[0].system.regret_per_round` | +0.00928 [+0.00743, +0.01113] | 是 |
| `cells[1].runs[0].system.mean_ineligible_fraction` | +0.03846 [+0.03103, +0.04589] | 是 |
| `cells[1].runs[0].system.ineligible_fraction_by_agent[0]` | +0.02170 [+0.01381, +0.02960] | 是 |
| `cells[1].runs[0].system.ineligible_fraction_by_agent[1]` | +0.05262 [+0.03928, +0.06595] | 是 |
| `cells[1].runs[0].system.ineligible_fraction_by_agent[2]` | +0.04106 [+0.02742, +0.05470] | 是 |
| `cells[1].runs[0].MDE_G` | 0.00374603 | 是 |
| `cells[1].runs[0].MDE_D` | 0.00368277 | 是 |
| `cells[1].runs[0].MDE_D_plus_h0` | 0.00368277 | 是 |
| `cells[1].runs[1].G_RL` | +0.01318 [+0.01011, +0.01626] | 是 |
| `cells[1].runs[1].D` | +0.01185 [+0.00841, +0.01530] | 是 |
| `cells[1].runs[1].D_plus_h0` | +0.01185 [+0.00841, +0.01530] | 是 |
| `cells[1].runs[1].system.regret_per_round` | +0.00934 [+0.00749, +0.01119] | 是 |
| `cells[1].runs[1].system.mean_ineligible_fraction` | +0.03856 [+0.03052, +0.04660] | 是 |
| `cells[1].runs[1].system.ineligible_fraction_by_agent[0]` | +0.02010 [+0.01273, +0.02747] | 是 |
| `cells[1].runs[1].system.ineligible_fraction_by_agent[1]` | +0.05853 [+0.04171, +0.07534] | 是 |
| `cells[1].runs[1].system.ineligible_fraction_by_agent[2]` | +0.03705 [+0.02359, +0.05052] | 是 |
| `cells[1].runs[1].MDE_G` | 0.00406277 | 是 |
| `cells[1].runs[1].MDE_D` | 0.00389834 | 是 |
| `cells[1].runs[1].MDE_D_plus_h0` | 0.00389834 | 是 |
| `cells[1].runs[2].G_RL` | +0.01087 [+0.00776, +0.01398] | 是 |
| `cells[1].runs[2].D` | +0.00954 [+0.00596, +0.01311] | 是 |
| `cells[1].runs[2].D_plus_h0` | +0.00954 [+0.00596, +0.01311] | 是 |
| `cells[1].runs[2].system.regret_per_round` | +0.01031 [+0.00834, +0.01227] | 是 |
| `cells[1].runs[2].system.mean_ineligible_fraction` | +0.04373 [+0.03559, +0.05187] | 是 |
| `cells[1].runs[2].system.ineligible_fraction_by_agent[0]` | +0.02010 [+0.01312, +0.02708] | 是 |
| `cells[1].runs[2].system.ineligible_fraction_by_agent[1]` | +0.07082 [+0.05360, +0.08804] | 是 |
| `cells[1].runs[2].system.ineligible_fraction_by_agent[2]` | +0.04026 [+0.02620, +0.05432] | 是 |
| `cells[1].runs[2].MDE_G` | 0.00380001 | 是 |
| `cells[1].runs[2].MDE_D` | 0.00354820 | 是 |
| `cells[1].runs[2].MDE_D_plus_h0` | 0.00354820 | 是 |
| `cells[1].honest_system.regret_per_round` | +0.00702 [+0.00529, +0.00876] | 是 |
| `cells[1].honest_system.mean_ineligible_fraction` | +0.03039 [+0.02404, +0.03674] | 是 |
| `cells[1].honest_system.ineligible_fraction_by_agent[0]` | +0.02331 [+0.01536, +0.03125] | 是 |
| `cells[1].honest_system.ineligible_fraction_by_agent[1]` | +0.02921 [+0.01916, +0.03927] | 是 |
| `cells[1].honest_system.ineligible_fraction_by_agent[2]` | +0.03866 [+0.02439, +0.05292] | 是 |
| `cells[1].G_HW.public` | +0.00133 [+0.00027, +0.00239] | 是 |
| `cells[1].G_HW.plus_h0` | +0.00133 [+0.00027, +0.00239] | 是 |
| `cells[1].G_HW.including_oracle` | +0.02111 [+0.01756, +0.02467] | 是 |
| `cells[1].hw_system[honest].regret_per_round` | +0.00702 [+0.00529, +0.00876] | 是 |
| `cells[1].hw_system[honest].mean_ineligible_fraction` | +0.03039 [+0.02404, +0.03674] | 是 |
| `cells[1].hw_system[honest].ineligible_fraction_by_agent[0]` | +0.02331 [+0.01536, +0.03125] | 是 |
| `cells[1].hw_system[honest].ineligible_fraction_by_agent[1]` | +0.02921 [+0.01916, +0.03927] | 是 |
| `cells[1].hw_system[honest].ineligible_fraction_by_agent[2]` | +0.03866 [+0.02439, +0.05292] | 是 |
| `cells[1].hw_system[dz=0.01].regret_per_round` | +0.00713 [+0.00536, +0.00891] | 是 |
| `cells[1].hw_system[dz=0.01].mean_ineligible_fraction` | +0.03093 [+0.02463, +0.03722] | 是 |
| `cells[1].hw_system[dz=0.01].ineligible_fraction_by_agent[0]` | +0.02331 [+0.01536, +0.03125] | 是 |
| `cells[1].hw_system[dz=0.01].ineligible_fraction_by_agent[1]` | +0.03082 [+0.02114, +0.04050] | 是 |
| `cells[1].hw_system[dz=0.01].ineligible_fraction_by_agent[2]` | +0.03866 [+0.02439, +0.05292] | 是 |
| `cells[1].hw_system[edge_oracle em=0.05].regret_per_round` | +0.00986 [+0.00796, +0.01175] | 是 |
| `cells[1].hw_system[edge_oracle em=0.05].mean_ineligible_fraction` | +0.04680 [+0.03802, +0.05558] | 是 |
| `cells[1].hw_system[edge_oracle em=0.05].ineligible_fraction_by_agent[0]` | +0.02090 [+0.01292, +0.02888] | 是 |
| `cells[1].hw_system[edge_oracle em=0.05].ineligible_fraction_by_agent[1]` | +0.08724 [+0.06928, +0.10520] | 是 |
| `cells[1].hw_system[edge_oracle em=0.05].ineligible_fraction_by_agent[2]` | +0.03225 [+0.01983, +0.04467] | 是 |
