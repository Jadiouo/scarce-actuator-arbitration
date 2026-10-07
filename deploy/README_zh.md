# 方向二：學校 GPU 機器部署手冊

給第一次做部署的同學。照順序做，每一步都有「預期會看到什麼」；看到的和預期不同就停下來，先查最後一節「常見問題」。
所有指令都在學校機器的終端機執行，`$` 代表提示符號，不用打。

整體流程：**帶程式過去 → 確認 GPU → 起環境（Docker 或 venv）→ 跑驗收 → 在 tmux 啟動執行器 → 設定 GitHub 備份 → 定期看進度 → 帶結果回來**。

> 重要：執行器會自動 `git commit`（每段一個），並依第 9 節設定自動 `git push` 到 GitHub 的 `direction2-freeze` 分支做備份。
> 備份只會推這一條分支，不會推 master，也不會 force push。

---

## 1. 把程式帶過去

### 1-A. 主要方式：git clone（學校機器能連 GitHub 時）

```bash
$ cd ~
$ git clone --branch direction2-freeze https://github.com/Jadiouo/scarce-actuator-arbitration.git
$ cd scarce-actuator-arbitration
$ git log --oneline | head -3
```

**預期**：clone 結束沒有 error；`git log` 第一行是最新的 Direction 2 commit；`git branch` 顯示 `* direction2-freeze`。
之後的備份設定（第 9 節）會把 `origin` 改成 SSH 位址，clone 時用 https 沒關係。

### 1-B. 離線替代方式：git bundle（不能連 GitHub 時）

在本機已經產生 `deploy/out/scarce-actuator-direction2.bundle`（含 direction2-freeze 完整歷史）。用 USB 或 scp 帶到學校機器，然後：

```bash
$ git bundle verify scarce-actuator-direction2.bundle
$ git clone --branch direction2-freeze scarce-actuator-direction2.bundle scarce-actuator-arbitration
$ cd scarce-actuator-arbitration
```

**預期**：`verify` 輸出含 `scarce-actuator-direction2.bundle is okay` 與 `The bundle records a complete history.`；clone 完成後 `git branch` 顯示 `* direction2-freeze`。
bundle clone 出來的 `origin` 指向 bundle 檔案，要備份到 GitHub 時照第 9 節改成 GitHub 位址即可。

### 1-C. 確認程式沒被改過

```bash
$ git status --short
$ git merge-base --is-ancestor a72d3d7 HEAD && echo "OK: 凍結點之後"
```

**預期**：`git status` 不列出任何東西；第二行印出 `OK: 凍結點之後`。若不是，執行器會拒絕開跑（這是故意的防呆）。

---

## 2. 檢查 GPU 與驅動

```bash
$ nvidia-smi
```

**預期**：看到一張表，有 GPU 名稱（目標機器是 Quadro RTX 8000）、`Driver Version`、`CUDA Version`、記憶體 48GB 左右。
驅動版本要 **>= 525.60**（越新越好；>= 550 最理想）。

- 指令不存在或印出 `couldn't communicate with the NVIDIA driver`：驅動沒裝或沒載入，找學校機器管理員，這一步沒過不要往下做。
- 同時有別人在用 GPU（表格下方 Processes 有別的程式）：先問清楚再跑，同一張卡不要同時跑兩個重工作。

---

## 3. 測試 Docker 能不能用 GPU（建議的環境）

```bash
$ docker --version
$ docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**預期**：第一行印出版本；第二行印出和第 2 節同樣的 nvidia-smi 表格。

- `permission denied ... docker.sock`：你不在 docker 群組。請管理員執行 `sudo usermod -aG docker $USER`，重新登入。或改用第 5 節的 venv。
- `could not select device driver "" with capabilities: [[gpu]]`：沒裝 NVIDIA Container Toolkit。請管理員安裝，或改用第 5 節。
- 下載 image 很慢或失敗：學校網路問題，改用第 5 節 venv 或請管理員協助。

---

## 4. Build image，並把 repo 掛進容器

```bash
$ cd ~/scarce-actuator-arbitration
$ docker build -t d2-runtime .
```

**預期**：幾分鐘（要下載 torch，約 2 GB），最後一行出現 `naming to docker.io/library/d2-runtime`，中間有一行
`torch 2.5.1+cu124 cuda 12.4`（build 時沒有 GPU，所以看不到 sm_75；第 6 節的驗收會在真正的 GPU 上檢查）。驅動比較舊（525 到 549）也可用，加上
`--build-arg CUDA_VERSION=12.1.1 --build-arg TORCH_CUDA=cu121`。

啟動容器（repo 掛在 `/work`，結果直接寫回學校機器上的資料夾，容器關掉也不會丟）：

```bash
$ docker run --rm -it --gpus all -u $(id -u):$(id -g) \
    -v "$PWD":/work -v ~/.ssh:/tmp/.ssh:ro d2-runtime bash
```

**預期**：提示符號變成 `I have no name!@xxxx:/work$`（正常，容器內沒有你的使用者名稱），`ls` 看得到 `scripts`、`arbitration`。

> 容器裡要 `git push` 備份時，需要 ssh 金鑰，細節見第 9 節的「在容器內推送」。如果覺得麻煩，可以不進容器跑執行器，
> 只用容器跑驗收；或直接用第 5 節 venv（備份最單純）。

驗證容器看得到 GPU：

```bash
(容器內) $ python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**預期**：`True Quadro RTX 8000`。印 `False`：回第 2、3 節。

---

## 5. 不能用 Docker：改用 venv

```bash
$ cd ~/scarce-actuator-arbitration
$ bash deploy/setup_venv.sh --dry-run     # 先看它打算裝什麼
$ bash deploy/setup_venv.sh
$ source .venv/bin/activate
```

**預期**：`--dry-run` 印出 `driver version`、`torch wheel`（例如 `torch==2.5.1 from .../whl/cu124`）後說 `(dry run: nothing installed)`。
正式執行幾分鐘後最後印 `OK. Next: source .venv/bin/activate && bash scripts/d2_target_check.sh`。
`source` 之後提示符號前面多出 `(.venv)`。

- 找不到 python3 或版本太舊：需要 Python 3.10 以上；`--python python3.11` 指定。
- 驅動太舊的警告（cu118）：照畫面說明，並記下來回報。

---

## 6. 執行驗收腳本

在容器內或 venv 內，於 repo 根目錄：

```bash
$ bash scripts/d2_target_check.sh --quick       # 先跑快速版（幾分鐘），確認腳本本身能跑
$ bash scripts/d2_target_check.sh               # 完整版，約 30 到 60 分鐘
```

**預期**：依序印出 `[1/6]` 到 `[6/6]`（環境、float64、CPU 測試、GPU 測試、速度、判定）。每一步結尾有 `exit code 0`。
最後一行：

```
可以開跑
```

並寫出 `results/direction2/target_check.json`，內含這台機器的每步耗時與 S1、主實驗的預估總時數。
`--quick` 的時數預估**不可信**，只看流程有沒有走完。

- 最後一行是 `不可以開跑`：往上看它列的原因（每個失敗步驟的 log 在 `results/direction2/target_check_work/`），不要硬跑。把 `target_check.json` 與 log 帶回來討論。
- 想先不碰 GPU 看一眼：`bash scripts/d2_target_check.sh --cpu-only`。

本機（RTX 5070 Ti）的對照數字請見本檔最後的「本機基準」。學校機器的預估時數若比本機的估計多 2 倍以上，先回報再開跑。

---

## 7. 在 tmux 或 nohup 啟動執行器

先看一下清單（不會執行任何東西）：

```bash
$ python3 scripts/d2_run_queue.py --dry-run | tail -3
```

**預期**：最後一行類似 `143 segments, 0 done, 143 to run; estimated 40h..m mean ...`。

啟動（推薦 tmux，斷線也不會停）：

```bash
$ tmux new -s d2
(tmux 內) $ python3 scripts/d2_run_queue.py 2>&1 | tee -a results/direction2/runlogs_console.txt
```

離開 tmux 但讓它繼續跑：按 `Ctrl+b` 再按 `d`。回來：`tmux attach -t d2`。

沒有 tmux 時用 nohup：

```bash
$ nohup python3 scripts/d2_run_queue.py > results/direction2/runlogs_console.txt 2>&1 &
$ echo $!        # 記下這個 PID
```

**預期**：開頭印 `0/143 segments already done; 143 to run`，接著每段一行 `START ...`，之後 `DONE ... committed <sha>`，
再接一行 `PUSH origin direction2-freeze: ok`（第 9 節設定好之後）。第一段通常需要 10 到 25 分鐘。

自動 push 選項：`--push-every N`（每 N 段推一次，預設 1）、`--remote`（預設 origin）、`--branch`（預設 direction2-freeze）、`--no-push`（不推）。
若目前所在分支不是 `--branch`，或 `--branch` 是 master／main，執行器會**拒絕啟動**並告訴你原因（exit code 3）；push 失敗則只記錄，不會中斷訓練，下一段完成時重試。

在 docker 內跑：先照第 4 節進容器，再執行同一行指令。

---

## 8. 查看進度、中斷與續跑

另開一個終端機：

```bash
$ python3 scripts/d2_run_queue.py --status
```

**預期**：每個 cell 一行（`done/seg` 與 `DONE / started / waiting`），接著 `segments done N/143`、「這台機器的速度是參考估計的幾倍」、
預估剩餘時間、最近三次執行紀錄、`runner: RUNNING`，以及 `unpushed commits (HEAD vs origin/direction2-freeze): 0`。
**未推送的 commit 數一直大於 0 且持續增加**代表備份壞了，看第 9 節與「常見問題」。

其他可看的東西：

```bash
$ tail -n 20 results/direction2/runlogs/push.log          # 每次 push 成功或失敗
$ tail -n 5  results/direction2/runlogs/queue_history.jsonl
$ nvidia-smi                                              # GPU 使用率應該很高
```

**中斷後續跑**：任何原因停了（斷電、Ctrl+C、被關機、出錯），**重新執行同一行指令**就好：

```bash
$ python3 scripts/d2_run_queue.py
```

**預期**：開頭 `K/143 segments already done`，完成的段會跳過，被中斷的那段從它的 checkpoint 接著跑，不會從頭開始。
想暫停時，在 tmux 裡按 `Ctrl+C` 一次，等它印出 `INTERRUPTED ... Nothing is lost`。
有一段失敗時，執行器會停下並印出原因、log 路徑與重跑指令；它**不會**跳過失敗的段。

---

## 9. GitHub 備份（deploy key 設定）

目標：每完成一段就把 commit 推到 GitHub 的 `direction2-freeze`，萬一學校機器壞掉也不會丟結果。
學校機器使用 **deploy key**（只對這一個 repo 有寫入權限，不是你的個人密碼）。

**步驟 1：在學校機器產生金鑰**

```bash
$ ssh-keygen -t ed25519 -f ~/.ssh/arb_deploy -N ""
$ cat ~/.ssh/arb_deploy.pub
```

**預期**：印出 `Your public key has been saved in ...arb_deploy.pub`；`cat` 印出一行 `ssh-ed25519 AAAA... 你的帳號@機器名`。
這行是**公鑰**，可以給人看；`~/.ssh/arb_deploy`（沒有 .pub）是私鑰，不要貼到任何地方。

**步驟 2：設定 ~/.ssh/config**

```bash
$ mkdir -p ~/.ssh && chmod 700 ~/.ssh
$ cat >> ~/.ssh/config <<'CFG'
Host github-arb
    HostName github.com
    User git
    IdentityFile ~/.ssh/arb_deploy
    IdentitiesOnly yes
CFG
$ chmod 600 ~/.ssh/config
```

**預期**：沒有輸出。

**步驟 3：請使用者（Jadiouo）把公鑰貼到 GitHub**

這一步要由 repo 擁有者操作，把步驟 1 印出的那一行公鑰傳給他：
GitHub 上開 `Jadiouo/scarce-actuator-arbitration` → **Settings** → **Deploy keys** → **Add deploy key** →
Title 填 `school-gpu`，Key 貼上公鑰，**勾選 Allow write access**，按 **Add key**。

**預期**：Deploy keys 清單出現 `school-gpu`，旁邊顯示 `Read/write`。沒勾 write 的話之後 push 會被拒絕。

**步驟 4：把 origin 改成 SSH 位址**

```bash
$ git remote set-url origin git@github-arb:Jadiouo/scarce-actuator-arbitration.git
$ git remote -v
```

**預期**：兩行 `origin  git@github-arb:Jadiouo/scarce-actuator-arbitration.git`（fetch 與 push）。

**步驟 5：驗證連線**

```bash
$ ssh -T git@github-arb
```

**預期**：第一次會問 `Are you sure you want to continue connecting (yes/no)?`，輸入 `yes`；
然後印 `Hi Jadiouo/scarce-actuator-arbitration! You've successfully authenticated, but GitHub does not provide shell access.`
（這句話是成功，指令的回傳碼 1 是正常的。）

- `Permission denied (publickey)`：公鑰沒貼對或貼到別的 repo；回步驟 3。
- `Could not resolve hostname`／逾時：學校網路擋 SSH。可改走 443 埠：在 config 的 `github-arb` 區塊把 `HostName github.com` 換成
  `HostName ssh.github.com` 並加一行 `Port 443`。

**步驟 6：確認這台機器的 git 身分**

```bash
$ git config user.name  || git config user.name "school-gpu"
$ git config user.email || git config user.email "school-gpu@localhost"
```

之後執行器的 commit 會用這個身分。

**步驟 7：（選用）手動推一次確認能寫入**

```bash
$ git push origin direction2-freeze
```

**預期**：`Everything up-to-date` 或顯示推了幾個物件。出現 `ERROR: The key you are authenticating with has been marked as read only`：
回步驟 3 勾選 Allow write access。**不要**加 `--force`，也不要推別的分支。

之後執行器每完成一段就自動 `git push origin direction2-freeze`（不帶任何 force 選項）。

**在容器內推送**：容器內的使用者沒有 `~/.ssh`。最簡單的作法是 **commit 在容器內、push 在容器外**：
執行器在容器內時加上 `--no-push`，另開一個學校機器上的終端機（不是容器內）定時執行
`git push origin direction2-freeze`（可以用 `watch -n 600 git push origin direction2-freeze`）。
若要在容器內自動推，需要把私鑰掛進去並設定 `GIT_SSH_COMMAND`（較複雜，不建議第一次部署就做）。使用 venv 沒有這個問題。

---

## 10. 結果如何帶回來

結果有兩份，擇一或兩者都做：

1. **從 GitHub 拉**（最簡單）：在本機

   ```bash
   $ git fetch origin direction2-freeze
   $ git log --oneline origin/direction2-freeze | head
   ```

   **預期**：看到一串 `Direction 2: <cell> run<k> g<a>-<b> (part + checkpoint)`。
2. **離線**：在學校機器上
   ```bash
   $ git bundle create ~/d2-results.bundle direction2-freeze
   $ git bundle verify ~/d2-results.bundle
   $ tar czf ~/d2-runlogs.tgz results/direction2/runlogs results/direction2/target_check.json
   ```
   把兩個檔案帶回來。本機 `git fetch d2-results.bundle direction2-freeze:school-direction2`。

只有 `results/direction2/parts/` 與 `results/direction2/ckpt/` 會被 commit；log 在 `runlogs/`。

---

## 11. 常見問題

| 現象 | 原因與做法 |
|---|---|
| `REFUSED: refusing to train: HEAD is not a descendant of the freeze commit a72d3d7` | 你不在 direction2-freeze 分支，或程式被改過。`git status`、`git branch` 檢查；不要自己改 `run_plan.json`、`FREEZE.md`。 |
| `REFUSED: refusing to push: the checked-out branch is ...` | 目前分支不是 direction2-freeze。`git checkout direction2-freeze`；或加 `--no-push` 先跑。 |
| `REFUSED: another d2_run_queue.py is already running` | 已經有一個執行器在跑（tmux 裡）。不要開第二個；用 `--status` 確認。 |
| 狀態列 `unpushed commits` 一直增加 | push 一直失敗。看 `runlogs/push.log` 最後幾行：`Permission denied` 回第 9 節步驟 3；`Could not resolve host` 是網路。訓練不受影響，網路恢復後下一段會自動補推。 |
| 一段失敗 `STOPPED` | 看畫面印出的 log 最後 15 行與 log 路徑。CUDA out of memory：確認沒有別人在用 GPU（`nvidia-smi`）。修好後重跑同一行指令。 |
| `CUDA error: no kernel image is available` | torch wheel 沒有這張卡（sm_75）的核心。用 cu121 或 cu124 wheel（Docker build-arg 或 `setup_venv.sh --cuda cu121`）。 |
| `nvidia-smi` 正常但 `torch.cuda.is_available()` 是 False | 容器沒帶 `--gpus all`，或驅動比 torch wheel 的 CUDA 新舊不合。換 cu121 版重做。 |
| 容器內 `fatal: detected dubious ownership` | image 已設定 `safe.directory '*'`；若仍出現，確認用的是 build 後的新 image。 |
| 容器內建立的檔案屬於 root | 啟動容器時漏了 `-u $(id -u):$(id -g)`。 |
| 驗收的 GPU 測試失敗 | 這代表新環境和方向一模擬器對不上（可能是 GPU 或 torch 版本差異）。**不要開跑**，把 `target_check_work/gpu_tests.log` 帶回來。 |
| 硬碟空間不夠 | `df -h .`；`results/direction2/` 全部跑完預期只有數十 MB，torch 與 image 約 8 GB。 |
| 機器重開機 | 執行器停了，什麼都沒丟。回到 repo，重開 tmux，重新執行第 7 節那一行即可（第 8 節）。 |

---

## 本機基準（RTX 5070 Ti，供比對）

`bash scripts/d2_target_check.sh` 在本機完整跑一次（2026-10-07，經 gpujob 排隊，總共約 19 分鐘），結果「可以開跑」：

| 步驟 | 耗時 |
|---|---|
| CPU 測試（136 passed, 31 deselected） | 131 秒 |
| GPU 測試（7 passed） | 310 秒 |
| 速度：main_M3C_r0.5 每步 | 約 1.06 ms（驗證 124 秒） |
| 速度：s1_D4_s0.02 每步 | 約 1.4 到 1.7 ms（驗證 131 秒） |
| 速度：s1_D2_s0.02 每步 | 約 1.22 ms（驗證 120 秒） |

預估時數（含 p95，不含每段 60 秒啟動）：S1 平均 40.7 小時（p95 42.5），主實驗平均 18.0 小時（p95 18.1），
S1 加主實驗（p95 加啟動）約 64 小時，約 2.7 天。學校機器的 `target_check.json` 若比這些數字大很多，先回報。
另：本機 7 個 GPU 測試（含 T59）單獨計時共 430 秒，其中 T59 最久（123 秒）。

容器（d2-runtime，CUDA 12.4 + torch 2.5.1）內的結果：`docker run --gpus all d2-runtime nvidia-smi` 正常顯示 RTX 5070 Ti，
`torch.cuda.is_available()` 為 True、arch 含 sm_75；容器內 CPU 測試 136 passed（約 3 分 40 秒）。
