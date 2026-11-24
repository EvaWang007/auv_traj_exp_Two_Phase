# NOAA Two-Phase AUV Trajectory Prediction

本公开仓库按用户指定，**仅保存代码、配置和说明**。
训练数据、历史实验结果、预测数组和模型检查点保留在本地，不随此次提交上传。
NOAA 海流来自真实观测；AUV 航迹、机载测量和控制记录为仿真。

## 正式训练配置

| 项目 | 配置 |
|---|---|
| 系统与环境 | Ubuntu 22.04、Python 3.10、PyTorch 2.7.1 + CUDA 11.8 |
| 最终模型 | LSTM、CPI-LSTM、PSO-CPI-LSTM，各从头训练 30 epoch |
| 数据划分 | 700/100/200 条训练/验证/测试任务，91,000/13,000/26,000 个窗口 |
| 历史与预测 | 120 步历史、80 步未来计划，采样周期 0.2 秒 |
| PSO 默认预算 | 6 粒子 × 4 次迭代，每候选在子集上训练 5 epoch；最终模型另训练 30 epoch |
| 模型选择 | 各自验证损失最低，三模型完成后统一冻结测试 |

第一阶段GPU训练已实现；第二阶段控制/采集/微调仍为原CPU版本，仅完整支持CPI结构。
本仓库准备时尚未启动正式30轮训练。旧试跑文件留在本地，不包含在本仓库。

## 克隆代码与导入数据

```bash
git clone https://github.com/EvaWang007/auv_traj_exp_Two_Phase.git
cd auv_traj_exp_Two_Phase
```

单独上传此前交付的 `auv_twophase_gpu_phase1_30ep.tar.gz` 训练包及 `.sha256` 文件到服务器。
例如将它们放在本仓库目录的上一级，然后执行：

```bash
# 只导入归档中的data文件，检查清单的SHA256；不覆盖已存在且不同的数据。
python server/import_data.py --archive ../auv_twophase_gpu_phase1_30ep.tar.gz
```

也可通过SSH/SFTP把本地完整 `data/` 目录复制到本仓库根目录。
数据清单：[server/data_manifest.json](server/data_manifest.json)；清单只有大小和校验值，没有数据内容。
当前约1.46GB数据不会通过 `git clone` 下载，GitHub上也未公开它们。

## 配置与启动服务器

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
bash server/setup_env.sh
conda activate auv_traj_gpu
nohup bash server/run_phase1.sh > phase1_launcher.log 2>&1 &
tail -f phase1_launcher.log
```

完整步骤、预算、恢复与输出位置见 [SERVER_PHASE1_30EP.txt](SERVER_PHASE1_30EP.txt)。
配置：[server/phase1_30ep.json](server/phase1_30ep.json)。
中断后重跑同一启动命令，从已完成epoch恢复；不要修改正在恢复的代码、数据或配置。
启动预检验证数据SHA256及实际CUDA前向/反向，通过后才进入正式训练。

## 训练后主动保存结果

`data/`、`results/`、`inspection/`默认忽略，不会因一般的 `git add .` 公开上传。
训练脚本不会自动推送任何结果。仅当你主动执行以下命令时，才会提交所选实验文件。

先在服务器设置Git身份，并配置可写本仓库的SSH/HTTPS凭据；不要把token写进代码。
配置了GitHub SSH公钥后可使用：

```bash
git remote set-url origin git@github.com:EvaWang007/auv_traj_exp_Two_Phase.git
git config user.name "EvaWang007"
git config user.email "98267661+EvaWang007@users.noreply.github.com"
```

预览指标、配置和对比图的文件清单，**不会提交或推送**：

```bash
python server/sync_results.py --run results/formal_phase1_30ep_seed42
```

确认这些结果可以公开后，主动执行：

```bash
python server/sync_results.py --run results/formal_phase1_30ep_seed42 --push
```

默认不上传检查点、预测/训练数组、日志或临时文件。
若以后明确决定公开备份检查点，先安装并初始化Git LFS，再显式选择：

```bash
git lfs install
python server/sync_results.py --run results/formal_phase1_30ep_seed42 --checkpoints --push
```

数组也必须显式加 `--arrays`。大数组和检查点按 `.gitattributes` 用LFS保存；
其存储、下载会占用仓库所有者的额度。初次代码提交没有上传任何LFS数据。
同步脚本只处理指定的 `results/` 子目录；已有暂存改动时停止，避免混入其他工作。
它不会强制推送或自动解决远程冲突。推送被拒绝时，本地提交仍保留，检查合并后再推送。
多服务器可使用不同分支，如 `git switch -c runs/server-a-seed42`。

## 代码结构

- `formal_phase1.py`：统一GPU训练、PSO与恢复入口。
- `formal_phase1_evaluate.py`：冻结测试、指标和出图。
- `phase1_model.py`：CPI网络、物理积分及原损失。
- `server/`：环境、正式配置、启动、数据导入与结果同步脚本、测试。
- `prepare_phase1.py`、`routes.py`：原数据生成与路线构造。
- `phase2_control.py`、`run_phase2.py`、`verify_phase2.py`：第二阶段原实现。
- 其他对比/试跑入口为历史研究脚本，运行时需本地相应数据或历史检查点。

输入与监督说明：[TRAINING_SPEC_REVIEW.txt](TRAINING_SPEC_REVIEW.txt)；
数组规格：[training_data_inventory.json](training_data_inventory.json)。
旧文档中的历史入口以新服务器说明为准；其中提及的本地结果不在公开代码仓库内。

## 评估边界

第一阶段离线遮蔽定位，历史动作仍来自正常定位控制器；第二阶段才采集模型控制的断联历史。
既有测试集曾用于试跑，新训练结果属于配对复测，不是新任务上的盲测。
窗口在集合内部重叠；单站、单种子结果不代表跨环境统计优势。

Git LFS：[官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、
[存储与带宽](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。
