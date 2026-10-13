#!/usr/bin/env python3
"""CPU-only report and independent arithmetic audit of stored post-main diagnostics."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/direction2"
GROUPS = ["current_type", "suspension", "recent_wins", "recent_reports", "residual", "slow_ema", "only_current_type"]
LABELS = ["當前型別", "暫停／資格", "近期得標", "近期報告偏移", "殘差", "慢速 EMA", "只保留當前型別"]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ci(a, ncomp=1):
    a = np.asarray(a, float)
    half = stats.t.isf(.025/ncomp, len(a)-1) * np.std(a, ddof=1) / math.sqrt(len(a))
    return dict(mean=float(np.mean(a)), lo=float(np.mean(a)-half), hi=float(np.mean(a)+half), half=float(half))


def gain(raw):
    u = np.asarray(raw["snap"]["util_b"], float)[..., 1]
    return (u-u[0])/raw["T_score"]


def f(d):
    return f"{d['mean']:+.6f} [{d['lo']:+.6f}, {d['hi']:+.6f}]"


def main():
    assertions = []

    def eq(a,b,name):
        assert np.allclose(a,b,rtol=1e-11,atol=1e-13), name
        assertions.append(name)

    def check_ci(got,a,name,ncomp=1):
        expected = ci(a,ncomp)
        for k in expected:
            eq(got[k],expected[k],name+"."+k)

    def raw(name):
        d = read(DATA/"postmain"/name)
        assert digest({k:v for k,v in d.items() if k!="sha256"}) == d["sha256"]
        assert d["contract"]["protocol_sha256"] == sha(ROOT/"docs/direction2/design/postmain-diagnostics.md")
        assert d["contract"]["diagnostic_script_sha256"] == sha(ROOT/"scripts/d2_postmain_diag.py")
        assert d["contract"]["seeds"] == list(range(2000,2000+len(d["contract"]["seeds"])))
        return d

    tails = read(DATA/"trunc_bias.json")
    tdocs = []
    for r in (.5,.9):
        name = f"main_M3C_r{r}"
        tr = raw(name+"__tail_raw.json")
        td = read(DATA/"postmain"/(name+"__tail.json"))
        tdocs.append(td)
        vr = read(DATA/"main_eval"/(name+"__val_rl_raw.json"))
        vh = read(DATA/"main_eval"/(name+"__val_hw_raw.json"))
        val = read(DATA/"main_eval"/(name+"__val.json"))
        selected = val["selected_run"]
        hi = vh["contract"]["policies"].index(val["selected_hw"]["public"])
        normal = [gain(vr)[selected+1],gain(vh)[hi]]
        variant = gain(tr)[1:]
        for j,row in enumerate(td["rows"]):
            assert row in tails
            eq(row["normal_per_seed"],normal[j],name+f".normal[{j}]")
            eq(row["variant_per_seed"],variant[j],name+f".variant[{j}]")
            delta = normal[j]-variant[j]
            eq(row["delta_per_seed"],delta,name+f".delta[{j}]")
            eq([row["delta"],*row["ci"]],[ci(delta)[k] for k in ("mean","lo","hi")],name+f".CI[{j}]")
            eq(row["G_normal"],normal[j].mean(),name+f".Gn[{j}]")
            eq(row["G_variant"],variant[j].mean(),name+f".Gv[{j}]")
            mde = val["runs"][selected]["mde_frozen"] if j==0 else val["hw_mde_frozen"][row["policy"]]
            eq(row["MDE_G_frozen"],mde,name+f".MDE[{j}]")
            assert row["flag_lower_rule"] == bool(ci(delta)["lo"]>0 and ci(delta)["lo"]>=mde)
            assert row["flag_mean_rule"] == bool(ci(delta)["lo"]>0 and ci(delta)["mean"]>=mde)
        dn,dv = normal[0]-normal[1],variant[0]-variant[1]
        check_ci(td["D_normal"],dn,name+".D_normal")
        check_ci(td["D_variant"],dv,name+".D_variant")
        check_ci(td["delta_D"],dn-dv,name+".delta_D")

    ar = raw("ablation_raw.json")
    ab = read(DATA/"postmain_ablation.json")
    assert ar["sha256"]==ab["raw_sha256"]
    val = read(DATA/"main_eval/main_M3C_r0.5__val.json")
    hand = read(DATA/"main_eval/main_M3C_r0.5__val_hw_raw.json")
    gh = gain(hand)[hand["contract"]["policies"].index(val["selected_hw"]["public"])]
    ga = gain(ar)
    for j,row in enumerate(ab["rows"]):
        i,k = row["run"],GROUPS.index(row["group"])
        orig,abl = ga[1+i*8],ga[2+i*8+k]
        d,drop = abl-gh,orig-abl
        eq(row["G_per_seed"],abl,f"ablation[{j}].G_raw")
        eq(row["D_per_seed"],d,f"ablation[{j}].D_raw")
        eq(row["drop_per_seed"],drop,f"ablation[{j}].drop_raw")
        check_ci(row["G"],abl,f"ablation[{j}].G")
        check_ci(row["D"],d,f"ablation[{j}].D")
        check_ci(row["drop"],drop,f"ablation[{j}].drop")
        check_ci(row["drop_simultaneous_95"],drop,f"ablation[{j}].drop_adjusted",21)
        assert row["material_dependency"] == bool(ci(drop,21)["lo"]>0 and drop.mean()>=.005)

    br = raw("behavior_raw.json")
    be = read(DATA/"postmain_behavior.json")
    trace = ROOT/be["trace"]
    assert sha(trace)==be["trace_sha256"]==br["trace_sha256"]
    assert be["reconstruction_bitwise_equal"] and br["reconstruction_bitwise_equal"]
    names = br["contract"]["names"]
    with np.load(trace,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    u,v,w,susp = [arrays[k][...,2500:17500] for k in ("u1","v1","won1","susp1")]
    delta = v-u
    el = np.arange(2501,17501)[None,None,:]>susp
    previous = arrays["won1"][...,2499:17499]
    for row in be["rows"]:
        i = names.index(row["policy"])
        g = row["group"]
        if g.startswith("u_bin_"):
            b=int(g.rsplit("_",1)[1]);mask=el[i] & (u[i]>=b/10) & (u[i]<(b+1)/10)
        else:
            mask={"eligible":el[i],"suspended":~el[i],"eligible_prev_win":el[i]&previous[i],"eligible_prev_loss":el[i]&~previous[i]}[g]
        assert row["count"]==int(mask.sum())
        vals=delta[i][mask]
        if len(vals):
            eq(row["mean_delta"],vals.mean(),f"behavior.{names[i]}.{g}.delta")
            eq(row["inflate_rate"],np.mean(vals>1e-6),f"behavior.{names[i]}.{g}.inflate")
            eq(row["underreport_rate"],np.mean(vals < -1e-6),f"behavior.{names[i]}.{g}.underreport")
        for j,x in enumerate(row["by_seed"]):
            vals=delta[i,j][mask[j]]
            assert x["count"]==len(vals)
            if len(vals):
                eq(x["mean_delta"],vals.mean(),f"behavior.{names[i]}.{g}.seed{j}.delta")
                eq(x["win_rate"],w[i,j][mask[j]].mean(),f"behavior.{names[i]}.{g}.seed{j}.win")

    # Algebraic inspection of the already-frozen networks, not an extra experiment.
    feature_names=["z","u","elig","rem","since"]+[f"won_{i}" for i in range(1,9)]+[f"d_{i}" for i in range(1,9)]+["res_last","ema_res","ema_d","ema_win"]
    structure=[]
    for i in range(3):
        ckpath=DATA/f"ckpt/main_M3C_r0.5__run{i}.npz"
        with np.load(ckpath,allow_pickle=False) as z:
            state=json.loads(str(z["state"]))
        theta=np.asarray(state["best"]["theta"],dtype="<f8")
        assert hashlib.sha256(theta.tobytes()).hexdigest()==val["checkpoints"][i]["theta_sha256"]
        prod=theta[:50].reshape(25,2)*theta[52:54]
        signs={n:("nondecreasing" if (p>=0).all() else "nonincreasing" if (p<=0).all() else "mixed") for n,p in zip(feature_names,prod)}
        structure.append(dict(run=i,theta_sha256=val["checkpoints"][i]["theta_sha256"],
                              hidden_weight_products=prod.tolist(),coordinate_monotonicity=signs))
    (DATA/"postmain_policy_structure.json").write_text(json.dumps(dict(schema="postmain-policy-structure/1",runs=structure),indent=2)+"\n")

    audit=dict(schema="postmain-number-audit/1",verified=True,assertions=len(assertions),checks=assertions,
               source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [DATA/"trunc_bias.json",DATA/"postmain_ablation.json",DATA/"postmain_behavior.json",trace]},
               method="Recomputed paired t intervals and Bonferroni intervals from utility snapshots; behavioral groups from full public logs.")
    (DATA/"postmain_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")

    out=["# 主實驗後診斷：截斷偏差、消融與報告行為", "",
         "本批在學校完成；使用已固定的 checkpoint、機制與手寫基準，沒有再訓練。全部資料來自 validation，"
         "且假說是在看過正式主實驗後提出，所以屬事後探索性診斷，不是新 holdout；不修改正式 test、ledger 或 J3。", "",
         "## 1. 尾端強制誠實的影響", "",
         "正常值取自已保存 validation 快照；變體只在 t>95000 強制誠實，記分至 97500。"
         "ΔG=G_normal−G_variant，CI 以 32 個環境 seeds 的配對差計算。", "",
         "| r | 政策 | 正常 G | 變體 G | ΔG [95% CI] | 凍結 G-MDE | 下界／均值門檻 |",
         "|---|---|---:|---:|---|---:|---|"]
    for row in tails:
        out.append(f"| {row['r']} | {row['policy']} | {row['G_normal']:.6f} | {row['G_variant']:.6f} | "
                   f"{row['delta']:+.6f} [{row['ci'][0]:+.6f}, {row['ci'][1]:+.6f}] | {row['MDE_G_frozen']:.6f} | "
                   f"{'過' if row['flag_lower_rule'] else '未過'}／{'過' if row['flag_mean_rule'] else '未過'} |")
    out += ["", "下界門檻為 CI_lo>0 且 CI_lo≥既有 G-MDE；較寬的均值門檻為 CI_lo>0 且 mean≥同一 MDE。"
            "MDE 沿用該政策的凍結值，不是這次 ΔG 的重新功效估計。兩個判準完整列出。", "",
            "| r | 正常 D [95% CI] | 尾端誠實 D [95% CI] | ΔD [95% CI] |", "|---|---|---|---|"]
    for r,td in zip((.5,.9),tdocs):
        out.append(f"| {r} | {f(td['D_normal'])} | {f(td['D_variant'])} | {f(td['delta_D'])} |")
    flagged=[f"r={r['r']} {r['policy']}" for r in tails if r['flag_lower_rule']]
    out += ["", ("依原規格字面門檻，須註明 G 含截斷偏差："+"、".join(flagged)) if flagged else
            "四項均未觸發原規格字面門檻。這不代表偏差為零；應同時看上表的 ΔG 量級與區間。", "",
            f"主要設定 r=0.5 的 D 從 {tdocs[0]['D_normal']['mean']:.6f} 變為 {tdocs[0]['D_variant']['mean']:.6f}，"
            f"變化量為 {f(tdocs[0]['delta_D'])}。"
            + ("尾端誠實後的 D 區間仍高於零，因此在本組 validation 上，這個尾端干預沒有消除學習策略的優勢。"
               if tdocs[0]['D_variant']['lo']>0 else "尾端誠實後的差距須連同區間判讀。"), "",
            "此量測同時移除了尾端攻擊收益與後續效果，只是規格指定的截斷偏差近似，不能精確等同「未收取的未來罰款」。"
            "ΔD 表示該干預對 RL 與手寫差距的影響，不回填或校正正式 test。", "",
            "## 2. 固定策略的特徵依賴", "",
            "每次把一組輸入的第一層權重設為零，其餘不變，不重訓。drop=G_original−G_ablation；"
            "正值代表原策略移除該組後變差。所有 21 個 drop 使用 Bonferroni 同時 95% 區間；"
            "下界>0 且均值≥0.005 才標為『實質依賴』。", "",
            "| run | 移除組別／變體 | G_ablation | D_ablation [名目 95% CI] | drop [同時 95% CI] | 實質依賴 |",
            "|---|---|---:|---|---|---|"]
    for row in ab["rows"]:
        out.append(f"| {row['run']} | {LABELS[GROUPS.index(row['group'])]} | {row['G']['mean']:+.6f} | {f(row['D'])} | "
                   f"{f(row['drop_simultaneous_95'])} | {'是' if row['material_dependency'] else '否'} |")
    common=[LABELS[k] for k,n in enumerate(GROUPS) if all(next(x for x in ab['rows'] if x['run']==i and x['group']==n)['material_dependency'] for i in range(3))]
    suspension=[x['drop']['mean'] for x in ab['rows'] if x['group']=='suspension']
    residual=[x['drop']['mean'] for x in ab['rows'] if x['group']=='residual']
    current_only=[x for x in ab['rows'] if x['group']=='only_current_type']
    out += ["", "三個 run 都達到實質依賴判準的干預："+("、".join(common) if common else "沒有共同項目")+"。", "",
            f"較有解釋力的是歷史狀態：移除暫停／資格資訊，三個 run 的 G 下降 {min(suspension):.4f}–{max(suspension):.4f}；"
            f"移除殘差資訊則下降 {min(residual):.4f}–{max(residual):.4f}。"
            + ("只保留當前型別時，三個 run 的 D 名目區間皆低於零；這些固定策略在此干預下都輸給手寫基準。"
               if all(x['D']['hi']<0 for x in current_only) else "只保留當前型別的結果見表。")
            + "當前型別被移除時跌幅最大，但這也是很強的分布外干預，不能把跌幅當成各特徵的通用重要性排序。", "",
            "這是在既有 checkpoint 中的功能依賴，不代表該資訊對任何可能策略都必要；零值干預可能離開訓練分布，"
            "各組有冗餘及非線性交互作用，drop 不能相加。未過門檻也不能宣稱該特徵無用。"
            "多重比較校正不消除 checkpoint 在同組 validation 選取的偏差。D 欄只有名目區間，不能當成新的正式發現。", "",
            "![消融影響](../../../figures/direction2-postmain-ablation.png)", "",
            "## 3. 原策略如何報告", "",
            "公開軌跡：每政策 8 個 validation seeds、20000 輪；以下取第 2501–17500 輪。"
            "數字描述觀察到的分布，不將每個時間點當獨立樣本。公開觀察全程可逐位元從公開日誌重建。", "",
            "| 策略 | 符合資格時平均 v−u | 誇大比例 | 縮小比例 | 暫停時平均 v−u |",
            "|---|---:|---:|---:|---:|"]
    for name in names:
        e=next(x for x in be['rows'] if x['policy']==name and x['group']=='eligible')
        s=next(x for x in be['rows'] if x['policy']==name and x['group']=='suspended')
        suspended_mean = f"{s['mean_delta']:+.5f}" if s['mean_delta'] is not None else "無樣本"
        out.append(f"| {name} | {e['mean_delta']:+.5f} | {e['inflate_rate']:.1%} | {e['underreport_rate']:.1%} | {suspended_mean} |")
    out += ["", "誇大／縮小用 |v−u|>1e−6 區分。資格、上一輪得標與型別區間的完整逐 seed 摘要均已保存。"
            "分組差異可能受型別分布及策略造成的歷史差異影響，單看曲線不能證明因果。", "",
            "| run | u<0.1 的平均 v−u | u≥0.9 的平均 v−u | u≥0.9 各 seed 均值範圍 | 上一輪得標／未得標後的平均 v−u |",
            "|---|---:|---:|---|---|"]
    for name in ('run0','run1','run2'):
        rows={x['group']:x for x in be['rows'] if x['policy']==name}
        high=[x['mean_delta'] for x in rows['u_bin_9']['by_seed'] if x['mean_delta'] is not None]
        out.append(f"| {name} | {rows['u_bin_0']['mean_delta']:+.5f} | {rows['u_bin_9']['mean_delta']:+.5f} | "
                   f"[{min(high):+.5f}, {max(high):+.5f}] | {rows['eligible_prev_win']['mean_delta']:+.5f}／"
                   f"{rows['eligible_prev_loss']['mean_delta']:+.5f} |")
    out += ["", "三個 run 的形狀接近：低急迫度時加得較多，高急迫度時收斂，在最高區間平均略微低報；"
            "後者在本批每個 seed 都同方向，但這裡沒有另作確認性檢定。上一輪得標後平均誇大較少，"
            "仍可能混有型別分布差異。綜合消融，較受支持的假說是：策略依當期型別決定基本偏移，再用暫停狀態與殘差調整報告。"
            "目前尚未拆解到各項調整如何產生淨收益，也沒有證明它重建了私有稽核分數。", "",
            "![原策略報告曲線](../../../figures/direction2-postmain-behavior.png)", "",
            "固定權重也可作純代數檢查。對輸出偏移 d=tanh(W2·h+b2)，"
            "∂d/∂x_i=(1−d²)Σ_j W2_j·(1−h_j²)·W1_ij。若兩個隱單元的 W2_j·W1_ij 同號，"
            "該輸入座標對 d 的方向在其他輸入固定時不會翻轉。以下直接由原 checkpoint 推得，不是新增模擬或顯著性檢定。", "",
            "| 輸入增加（其餘座標固定） | run0 的 d | run1 的 d | run2 的 d |", "|---|---|---|---|"]
    translations={"nondecreasing":"不減","nonincreasing":"不增","mixed":"方向視狀態而定"}
    for feat,label in [("elig","服務資格 0→1"),("since","距上次暫停通知時間"),("d_1","上一輪 v−u"),("res_last","上一輪得標後的殘差特徵"),("ema_win","得標的慢速 EMA"),("u","當前 u 特徵（固定 z）")]:
        out.append("| "+label+" | "+" | ".join(translations[s["coordinate_monotonicity"][feat]] for s in structure)+" |")
    out += ["", "這些方向描述的是 d，實際報告還有 v=clip(u+d)。輸入彼此會相關（尤其 z 與 u），"
            "固定其他座標不是實際環境一定會發生的情況；不能由此直接推論防守參數調整後的因果效果。"
            "全特徵係數與方向見 [policy_structure](../../../results/direction2/postmain_policy_structure.json)。", "",
            "## 4. 邊界與驗證", "",
            "S1 的基礎優勢混淆與主 run0 的先前 test 暴露仍然存在，本批不能修復這些限制。"
            "本批也未執行全新 holdout 或 E1 防守方學習；下一個確認性實驗須另行登記。", "",
            f"診斷前 5 項 CPU、1 項 GPU 驗收通過；GPU 驗收涵蓋遮罩 graph/eager、尾端干預邊界、記錄開關與公開觀察重建。"
            f"本報告另從快照與公開軌跡完成 {len(assertions)} 項數值核對。首次驗收抓到權重 view 的原地修改，"
            "在任何診斷量測前修正為複製權重，原 checkpoint 不變。第一個尾端變體曾因既有 ad_h=Infinity 的 JSON 設定資訊序列化失敗，"
            "該次沒有保存結果；改為只在設定資訊中明確編碼無限值、保留原模擬參數後，重跑 validation 變體。", "",
            "資料：[截斷偏差](../../../results/direction2/trunc_bias.json)、[消融](../../../results/direction2/postmain_ablation.json)、"
            "[行為](../../../results/direction2/postmain_behavior.json)、[數字稽核](../../../results/direction2/postmain_audit.json)、"
            "[交付驗證](../../../results/direction2/postmain_validation.json)、"
            "[原始快照與軌跡](../../../results/direction2/postmain/)。", "",
            "[執行前診斷協定](../design/postmain-diagnostics.md) · [正式主實驗結果](main.md)。"
            "重建報告：`.venv/bin/python scripts/report_d2_postmain.py`（CPU）。"]
    outpath=ROOT/"docs/direction2/results/postmain-diagnostics.md"
    outpath.write_text("\n".join(out)+"\n")

    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".scratch/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4.3))
    mat=np.array([[next(x for x in ab['rows'] if x['run']==i and x['group']==g)['drop']['mean'] for g in GROUPS] for i in range(3)])
    vmax=max(.005,np.abs(mat).max())
    im=ax.imshow(mat,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
    for i in range(3):
        for j,g in enumerate(GROUPS):
            row=next(x for x in ab['rows'] if x['run']==i and x['group']==g)
            ax.text(j,i,f"{mat[i,j]:+.4f}"+(" *" if row['material_dependency'] else ""),ha="center",va="center",fontsize=10,
                    color="white" if abs(mat[i,j])>.6*vmax else "black")
    ax.set_xticks(range(7),["Current\ntype","Suspension","Recent\nwins","Recent\nreports","Residual","Slow\nEMA","Only current\ntype"])
    ax.set_yticks(range(3),["run0","run1","run2"])
    ax.set_title("Fixed-policy ablations on validation: original G - ablated G",fontweight="bold")
    fig.colorbar(im,ax=ax,label="Gain lost after intervention")
    fig.text(.5,.025,"* Bonferroni simultaneous CI lower > 0 and mean drop >= 0.005. No retraining; post-hoc diagnostic.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(ROOT/"figures/direction2-postmain-ablation.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    colors=["#64748b","#2563eb","#d97706","#059669","#9333ea"]
    for name,col in zip(names,colors):
        y=[next(x for x in be['rows'] if x['policy']==name and x['group']==f'u_bin_{j}')['mean_delta'] for j in range(10)]
        axes[0].plot(np.arange(.05,1,.1),y,marker='o',label=name,color=col,linewidth=1.5)
    axes[0].axhline(0,color="#94a3b8",linewidth=1)
    axes[0].set(xlabel="Own true urgency u (bin center)",ylabel="Mean report shift v - u",title="Eligible rounds: reporting profile")
    axes[0].legend(fontsize=8)
    for i,(name,col) in enumerate(zip(names,colors)):
        row=next(x for x in be['rows'] if x['policy']==name and x['group']=='eligible')
        axes[1].barh(i-.16,row['inflate_rate'],height=.3,color=col)
        axes[1].barh(i+.16,-row['underreport_rate'],height=.3,color=col,alpha=.45)
    axes[1].set_yticks(range(len(names)),names);axes[1].invert_yaxis()
    axes[1].set(xlim=(-1,1),xlabel="Underreport fraction < 0 | Inflate fraction > 0",title="Eligible rounds: direction of reporting")
    axes[1].axvline(0,color="#64748b",lw=1)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.text(.5,.015,"8 validation seeds; rounds 2501-17500. Descriptive public-log summaries; these are not independent time samples.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(ROOT/"figures/direction2-postmain-behavior.png",dpi=180);plt.close(fig)
    print(json.dumps(dict(audit_checks=len(assertions),common_material_dependencies=common,tail_flags=flagged),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
