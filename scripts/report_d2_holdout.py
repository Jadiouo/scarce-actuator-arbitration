#!/usr/bin/env python3
"""Independent CPU arithmetic audit and report, using only stored holdout snapshots."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results/direction2/holdout"
LABELS={"confirmed_material_advantage":"確認實質優勢", "confirmed_positive_only":"確認正優勢，但未確認超過實質門檻",
        "material_advantage_ruled_out":"排除達到指定門檻的優勢", "inconclusive":"證據不足以判定"}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def ci(a,family=1):
    a=np.asarray(a,float)
    half=float(stats.t.isf(.025/family,len(a)-1)*np.std(a,ddof=1)/math.sqrt(len(a)))
    mu=float(np.mean(a))
    return dict(mean=mu,lo=mu-half,hi=mu+half,half=half)


def fmt(x):return f"{x['mean']:+.6f} [{x['lo']:+.6f}, {x['hi']:+.6f}]"


def main():
    m=read(DATA/"manifest.json");r=read(DATA/"result.json");ledger=read(DATA/"ledger.json")
    assert m['sha256']==digest({k:v for k,v in m.items() if k!='sha256'})
    assert r['manifest_sha256']==m['sha256'] and r['selected_run']==m['selected_run']==0
    assert r['n_seeds']==len(m['seeds'])==64 and r['delta']==m['delta']==.005
    for p,h in {**m['inputs_sha256'],**m['protected_sha256']}.items():assert sha(ROOT/p)==h,p
    sa=read(DATA/'seed_audit.json')
    assert sha(DATA/'seed_audit.json')==m['seed_audit_sha256'] and sa['new_base_seeds']==m['seeds']
    old={s+o for s in sa['historical_seed_field_values'] for o in (0,1000000,2000000,3000000)}
    fresh={s+o for s in m['seeds'] for o in m['generator_offsets']}
    assert len(fresh)==192 and not fresh&old
    original_ledger=read(ROOT/'results/direction2/final_eval_ledger.json')
    assert len(original_ledger)==42
    checks=[]
    def equal(a,b,name):
        assert np.allclose(a,b,rtol=1e-11,atol=1e-13),name
        checks.append(name)
    def check_ci(got,a,name,family=1):
        expected=ci(a,family)
        for k in expected:equal(got[k],expected[k],name+'.'+k)
    batches={}
    assert set(ledger)=={'rl','hw'}
    for name,npoli in [('rl',4),('hw',2)]:
        p=DATA/(name+'_raw.json');d=read(p)
        assert d['sha256']==digest({k:v for k,v in d.items() if k!='sha256'})==r['raw_sha256'][name]
        expected=dict(manifest_sha256=m['sha256'],stage=name,seeds=m['seeds'],n_policies=npoli,T_score=97500)
        assert d['contract']==expected and d['T_score']==97500
        assert ledger[name]['status']=='completed' and ledger[name]['raw_sha256']==sha(p)
        assert ledger[name]['contract_sha256']==digest(expected)
        for k,shape in [('util_b',(npoli,64,3)),('reg_b',(npoli,64)),('pen',(npoli,64,3))]:
            a=np.asarray(d['snap'][k],float)
            assert a.shape==shape and np.isfinite(a).all(),(name,k)
        batches[name]=d
    freeze_commit=subprocess.check_output(['git','log','-1','--format=%H','--','results/direction2/holdout/manifest.json'],cwd=ROOT,text=True).strip()
    freeze_time=int(subprocess.check_output(['git','show','-s','--format=%ct',freeze_commit],cwd=ROOT,text=True))
    assert freeze_time<=min(x['started_unix'] for x in ledger.values())
    rl,hw=batches['rl'],batches['hw']
    for k in ('util_b','reg_b','pen'):equal(rl['snap'][k][0],hw['snap'][k][0],'honest.'+k)
    ur=np.asarray(rl['snap']['util_b'])[:,:,1]
    uh=np.asarray(hw['snap']['util_b'])[:,:,1]
    gh=(uh[1]-uh[0])/97500
    check_ci(r['G_HW'],gh,'G_HW');equal(r['G_HW_per_seed'],gh,'G_HW.per_seed')
    for i,row in enumerate(r['runs']):
        assert row['run']==i
        g=(ur[i+1]-ur[0])/97500
        # Direct difference of policy utilities, independently of stored G vectors.
        d=(ur[i+1]-uh[1])/97500
        equal(row['G_per_seed'],g,f'run{i}.G_vector')
        equal(row['D_per_seed'],d,f'run{i}.D_vector')
        check_ci(row['G'],g,f'run{i}.G')
        check_ci(row['D'],d,f'run{i}.D')
        check_ci(row['D_simultaneous_95'],d,f'run{i}.D_family',3)
    systems=[('honest',rl,0,r['honest_system']),('handwritten',hw,1,r['handwritten_system'])]
    systems += [(f'run{i}',rl,i+1,x['system']) for i,x in enumerate(r['runs'])]
    for name,raw,j,row in systems:
        reg=np.asarray(raw['snap']['reg_b'])[j]/97500
        pen=np.asarray(raw['snap']['pen'])[j]/97500
        check_ci(row['regret_per_round'],reg,name+'.regret')
        check_ci(row['mean_ineligible_fraction'],pen.mean(-1),name+'.ineligible')
        for a in range(3):check_ci(row['ineligible_fraction_by_agent'][a],pen[:,a],name+f'.ineligible{a}')
    primary=ci((ur[1]-uh[1])/97500)
    if primary['lo']>.005:verdict='confirmed_material_advantage'
    elif primary['lo']>0:verdict='confirmed_positive_only'
    elif primary['hi']<.005:verdict='material_advantage_ruled_out'
    else:verdict='inconclusive'
    assert r['verdict']==verdict
    assert r['positive_advantage_ruled_out']==bool(primary['hi']<=0)
    k=sum(ci((ur[i+1]-uh[1])/97500,3)['lo']>0 for i in range(3))
    assert r['consistent_runs']==k
    exitfile=ROOT/'.scratch/holdout_20261013.exit'
    log=ROOT/'results/direction2/runlogs/holdout_20261013.log'
    receipt_path=DATA/'execution.json'
    if not receipt_path.exists():
        assert exitfile.read_text().strip()=='0' and '8 passed' in log.read_text()
        receipt=dict(schema='holdout-execution/1',manifest_sha256=m['sha256'],exit_code=0,
                     preflight=dict(cpu_passed=7,gpu_passed=1),log_sha256=sha(log),
                     batch_file_sha256={name:sha(DATA/(name+'_raw.json')) for name in ('rl','hw')})
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    receipt=read(receipt_path)
    assert receipt['manifest_sha256']==m['sha256'] and receipt['exit_code']==0
    assert receipt['preflight']==dict(cpu_passed=7,gpu_passed=1)
    for name,h in receipt['batch_file_sha256'].items():assert sha(DATA/(name+'_raw.json'))==h
    audit=dict(schema='independent-holdout-audit/1',verified=True,checks=checks,numeric_checks=len(checks),
               source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [DATA/(x+'.json') for x in ('manifest','seed_audit','rl_raw','hw_raw','ledger','result','execution')]},
               freeze_commit=freeze_commit,committed_before_first_attempt=True,old_test_ledger_entries=42,
               frozen_inputs_unchanged=True,new_generator_streams=192,preflight=dict(cpu_passed=7,gpu_passed=1),
               attempts=dict(rl=1,hw=1),exit_code=0,verdict=verdict,consistent_runs=k,
               method='Independent scipy t.isf computation from raw paired policy utilities; no evaluator import or simulation.')
    (DATA/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')

    out=['# 獨立環境 holdout 結果','',f"本批結論：**{LABELS[verdict]}**。固定主 run0 的 D={fmt(primary)}；"
         f"事前實質門檻 δ=0.005，判準為配對 95% CI 下界>δ。三個既有 run 中 {k}/3 的 Bonferroni 同時 95% CI 下界>0。",'',
         '本批全部在學校執行，模型、機制及手寫基準先凍結；64 個新環境 seeds 不參與訓練、選模或除錯。'
         '主設定 r=0.5，T=100000、T_score=97500；G 是相對誠實報告的每輪個人效用增量，D 是學習策略相對固定手寫 dz=0.3 的差。','',
         '| run | G [名目 95% CI] | D [名目 95% CI] | D [三個 run 同時 95% CI] |',
         '|---|---|---|---|']
    for row in r['runs']:
        out.append(f"| {row['run']}"+('（固定主值）' if row['run']==0 else '')+f" | {fmt(row['G'])} | {fmt(row['D'])} | {fmt(row['D_simultaneous_95'])} |")
    out += ['',f"手寫基準 G={fmt(r['G_HW'])}。每個 CI 的樣本單位是 64 個環境 seeds；"
            '三個 run 共用環境亂數，沒有合併成 192 個獨立樣本。次要同時區間描述固定 checkpoints 的一致性，不影響主判定。','',
            '![獨立 holdout 配對差值](../../../figures/direction2-independent-holdout.png)','',
            '## 系統代價（描述性）','',
            '| 政策 | regret／輪 [95% CI] | agent 0 不合格比例 | agent 1 不合格比例 | agent 2 不合格比例 |',
            '|---|---|---:|---:|---:|']
    for name,_,_,s in systems:
        out.append(f"| {name} | {fmt(s['regret_per_round'])} | "+' | '.join(f"{v['mean']:.5f}" for v in s['ineligible_fraction_by_agent'])+' |')
    out += ['', '上表只作系統結果的描述，沒有新增 regret 或不合格比例的顯著性主張。攻擊者增加私利不能解讀為系統整體改善。','',
            '## 規劃、執行與停止','',
            f"協定與 manifest 在首次 holdout 執行前提交（`{freeze_commit[:7]}`）。掃描 {sa['files_scanned']} 份原有 JSON／NPZ 的 seed 欄位，"
            '新三流共 192 個 generator seeds 與已保存 seed 值及其 offsets 無交集。此查核限本專案保存紀錄，無法證明外部未留紀錄的執行不存在。','',
            f"原規劃的 95% CI 半寬為 {m['planning']['expected_half_width_95']:.6f}，實際主值半寬為 {primary['half']:.6f}。"
            '新資料變異高於規劃假設；沒有依此追加樣本或調整門檻。'
            f"RL 批次 {rl['wall_s']:.1f} 秒、手寫批次 {hw['wall_s']:.1f} 秒；各執行一次，沒有中途更換模型、加 seeds 或重跑挑結果。"
            f"7 項 CPU 與 1 項 GPU 驗收通過，獨立 CPU 稽核完成 {len(checks)} 項數值核對。原正式 test ledger 仍為 42 筆；本批另有兩批次 ledger。",'',
            '固定樣本量已完成，這輪確認實驗到此停止。結論針對已固定的策略與機制；未再訓練，因此不估計新訓練的成功率。'
            '它補上新環境資料的確認，保留原 test 暴露與 S1 標定混淆的揭露，不能用來宣稱已找到最強攻擊或勝過所有手寫方法。','',
            '資料：[manifest](../../../results/direction2/holdout/manifest.json)、[逐 seed 結果](../../../results/direction2/holdout/result.json)、'
            '[獨立稽核](../../../results/direction2/holdout/audit.json)、[執行紀錄](../../../results/direction2/holdout/execution.json)、'
            '[原始快照／ledger](../../../results/direction2/holdout/)。','',
            '[執行前協定](../design/independent-holdout.md) · [本階段結論](conclusion.md) · [原正式 test](main.md) · [事後診斷](postmain-diagnostics.md)。'
            '重建：`.venv/bin/python scripts/report_d2_holdout.py`（CPU；已保存執行紀錄，可直接從快照重算）。']
    (ROOT/'docs/direction2/results/independent-holdout.md').write_text('\n'.join(out)+'\n')

    oldmain=read(ROOT/'results/direction2/main.json')
    oldcell=next(c for c in oldmain['cells'] if c['r']==.5)
    old=next(x for x in oldcell['runs'] if x['run_id']==oldcell['selected_run'])['D']
    tail=read(ROOT/'results/direction2/postmain/main_M3C_r0.5__tail.json')
    interpretation={
        'confirmed_material_advantage':'在固定 M3C、r=0.5 下，這個只看公開資訊的學習策略，相對固定手寫基準取得了超過事前門檻的額外私利；新環境 holdout 支持這個有限範圍的結論。',
        'confirmed_positive_only':'新環境 holdout 支持固定學習策略有正的額外私利，但尚未確認優勢超過每輪 0.005；實質優勢主張須保留。',
        'material_advantage_ruled_out':'新環境 holdout 排除了固定主策略達到每輪 0.005 的額外優勢；原 test 的實質優勢未獲本批確認。',
        'inconclusive':'新環境 holdout 仍不足以判定固定主策略是否有額外優勢；目前不能宣稱完成實質優勢的確認。'}[verdict]
    conclusion=['# 方向二：本階段結論與收尾','',interpretation,'',
                f"主要數字：**D={fmt(primary)}**，64 個新 seeds；事前門檻 δ=0.005，結果為 **{LABELS[verdict]}**。"
                f"三個既有 run 的次要同時區間有 {k}/3 高於零，表示這三個固定模型的方向一致性。",'',
                '| 證據 | 主 run0 的 D [95% CI] | 解讀範圍 |','|---|---|---|',
                f"| 原正式 test | {fmt(old)} | 原 J3 保留；run0 曾在 S1 使用相同 test |",
                f"| validation 尾端誠實診斷 | {fmt(tail['D_variant'])} | 事後診斷，非獨立確認 |",
                f"| 新環境 holdout | {fmt(primary)} | 策略與判準先固定的新環境確認 |",'',
                '機制解釋目前到這裡：既有消融顯示，三個固定策略都依賴當期型別、停權／資格與殘差資訊；'
                '低急迫度時加報較多，最高急迫度區間平均略微低報。這支持狀態相關的報告策略，但還沒有把各項調整如何產生淨收益完整拆開。','',
                '## 什麼已完成','',
                '| 工作 | 狀態 |','|---|---|','| 原正式 test、配對 CI 與登記複現判定 | 完成 |',
                '| 尾端影響、固定策略消融、公開行為分析 | 完成 |','| 獨立環境 holdout、數字稽核與本階段結論 | 完成；固定樣本量已用完 |',
                '| E1 防守方學習／機制修補 | 未開始；另立研究問題 |','',
                '本階段已可形成研究結論並整理論文。結論限制為這個固定 M3C、固定公開資訊模型與固定手寫基準；'
                '不代表所有機制、所有手寫策略或最強攻擊，也不代表新訓練的成功率。S1 的基礎優勢混淆仍是限制。','',
                '若接著研究防守，應另訂「哪項機制修改能降低攻擊者增益，同時不惡化誠實使用者的服務」的目標與門檻。'
                '那是下一個問題，不是這次結論的無限延長；這批不自動開始訓練。','',
                '[完整獨立 holdout 結果](independent-holdout.md) · [執行前協定](../design/independent-holdout.md) · [事後診斷](postmain-diagnostics.md) · [原正式 test](main.md)']
    (ROOT/'docs/direction2/results/conclusion.md').write_text('\n'.join(conclusion)+'\n')

    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.scratch/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for i,row in enumerate(r['runs']):
        ax.errorbar(row['D']['mean'],i,xerr=row['D_simultaneous_95']['half'],fmt='none',color='#94a3b8',lw=6,alpha=.6)
        ax.errorbar(row['D']['mean'],i,xerr=row['D']['half'],fmt='o',color='#2563eb',capsize=5,lw=2)
    ax.axvline(0,color='#64748b',lw=1)
    ax.axvline(.005,color='#be123c',ls='--',label='Preregistered material threshold: 0.005')
    ax.set_yticks(range(3),['run0 (primary)','run1','run2']);ax.invert_yaxis()
    ax.set_ylim(2.6,-.6)
    ax.set_xlabel('Paired gain difference: learned - fixed handwritten')
    ax.set_title('Independent environment holdout: 64 new seeds',fontweight='bold')
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='lower right')
    fig.text(.5,.025,'Blue: nominal paired 95% CI. Gray: simultaneous 95% CI over 3 fixed runs.\nPrimary decision uses run0 nominal CI lower > 0.005. No retraining or sample extension.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(ROOT/'figures/direction2-independent-holdout.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(verdict=verdict,primary_D=primary,consistent_runs=k,numeric_checks=len(checks)),indent=2))


if __name__=='__main__':main()
