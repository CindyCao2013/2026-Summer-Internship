# AlphaNet 复现与优化（v1 / v2 / v3）

华泰证券两篇报告的端到端复现：

1. 《AlphaNet：因子挖掘神经网络》（2020-06-14）→ **v1**  
   规范：[`docs/00_alphanet_reproduction_guide.md`](docs/00_alphanet_reproduction_guide.md)
2. 《再探 AlphaNet：结构和特征优化》（2020-08-24）→ **v2 / v3**  
   规范：[`docs/01_alphanet_v2_v3_guide.md`](docs/01_alphanet_v2_v3_guide.md)

输出目录：`research/results/alphanet_v1/`

实现后端是 **PyTorch**（层语义对齐指南中的 Keras 草图）。

## 结构对照

| | v1 | v2 | v3 |
|--|----|----|----|
| 输入 | 9 × 30 | **15 × 30**（+6 个比率） | 15 × 30 |
| 提取层 | 10 算子，d=10 | 10 算子，d=10 | **两层**：d=10 与 d=5，各 6 算子 |
| 中间层 | 池化 + Dense(30, ReLU) + Dropout 0.5 | **LSTM**(time_step=3, hidden=30) + BN | **GRU**(t=3) + **GRU**(t=6)，拼接 |
| 优化器 | RMSprop 1e-4 | Adam 1e-4 | Adam 1e-4 |
| 训练/验证 | 1:1 | **4:1** | 4:1 |
| batch | 1000 | 2000 / 800 / 500 | 500（中证500） |
| 增强双边换手上限 | 30% | **60%** | 60% |

v2/v3 的 6 个比率：`close/free_turn`、`open/turn`、`volume/low`、`vwap/high`、`low/high`、`vwap/close`。配对默认用 `C(F,2)`（v1 仍按研报用 9×9=81）。

## 修改后的测试协议

10 层（G1=Top）、全市场等权超额、单边成本 **万分之 7.5（7.5 bps，`0.00075`）**、五因子中性化。与仓库 L2 / `candidate_pool_v1` 回测口径一致。原报告 RankIC / 增强超额见进阶指南第 7 节；那些数字是千分之二或未扣费口径，本项目扣 7.5 bps 后绝对值会更低。不要把万分之 7.5 写成 7.5‰（千分之 7.5 = 0.75%，偏大 10 倍）。

## 变体

| 名称 | 含义 |
|------|------|
| `v1` | 论文 V1 |
| `v2` / `v2_csi800` / `v2_csi500` | V2，全 A / 中证800 / 中证500 |
| `v3` / `v3_alla` | V3，默认中证500 |
| `smoke` / `smoke_v2` / `smoke_v3` | CPU 冒烟 |
| `v1_adam` 等 | 见 `alphanet/variants.py` |

## 运行

```bash
PY=/opt/conda/anaconda3/bin/python
cd /path/to/factor_dev
PYTHONPATH=. $PY alphanet/scripts/run_smoke.py
PYTHONPATH=. $PY -m pytest -q alphanet/tests

PYTHONPATH=. $PY alphanet/scripts/run_train.py --variant v2
PYTHONPATH=. $PY alphanet/scripts/run_train.py --variant v3
PYTHONPATH=. $PY alphanet/scripts/run_evaluate.py --variant v2

# AlphaNet vs 日频显式因子（candidate_pool_v1 代表因子 + Size/Mom/Vol/Turnover）
PYTHONPATH=. $PY alphanet/scripts/run_compare_factors.py --variant v1
PYTHONPATH=. $PY alphanet/scripts/run_compare_factors.py --synthetic
```

对比报告写入 `research/results/alphanet_vs_explicit/`。候选池若没有 `factor_narrow.parquet`，仍会对经典风格因子做全样本截面相关与残差 IC，并在报告中标明池文件缺失。`--synthetic` 只用于打通框架，不能当作训练后的 AlphaNet 结论。

`base_93` 没有 torch，请用上面的 `$PY`。`run_train.py --n-seeds` 默认 10，并行上限 10。
