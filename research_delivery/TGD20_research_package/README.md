# TGD20 完整研究交付包

这是 TGD20 唯一的集中交付目录。因子定义、代码快照、缓存、回测产物、扩展诊断、图表和最终报告都在本目录内。

## 首要入口

- **现行报告**：[`report/TGD20_因子研究报告_v3.md`](report/TGD20_因子研究报告_v3.md) · [HTML](report/TGD20_因子研究报告_v3.html)
- 旧买方长报告（归档）：[`report/TGD20_买方因子研究报告.md`](report/TGD20_买方因子研究报告.md)
- Mentor 协议数据：[`data/analysis/mentor_protocol/`](data/analysis/mentor_protocol/)
- 协议 runner：仓库根目录 `run_mentor_single_factor_protocol.py`
- 因子公式：[`specification/factor_specs/TGD20.yaml`](specification/factor_specs/TGD20.yaml)
- 核心实现：[`code/canonical/core/l2_features/`](code/canonical/core/l2_features/)
- 正式验证：[`code/canonical/run_tgd_validation_v1.py`](code/canonical/run_tgd_validation_v1.py)
- 买方扩展诊断：[`scripts/build_tgd20_buyside_diagnostics.py`](scripts/build_tgd20_buyside_diagnostics.py)
- 文件血缘：[`SOURCE_MANIFEST.csv`](SOURCE_MANIFEST.csv)
- 全包校验：[`PACKAGE_INVENTORY.csv`](PACKAGE_INVENTORY.csv)

## 目录

```text
TGD20_research_package/
├── report/                 # 最终买方研究报告
├── specification/          # 冻结定义、叙事 schema、里程碑说明
├── code/
│   ├── canonical/          # TGD 专属源码与 runner 快照
│   └── dependencies/       # 回测所需框架依赖快照
├── data/
│   ├── cache/              # 可离线重建回测的 TGD 中间层与宽/长表
│   ├── analysis/           # 买方诊断数据
│   └── DATA_LINEAGE.md     # 原始数据、复权与时点口径
├── artifacts/              # 历史正式验证、Golden Pack、正交化产物
├── figures/                # 最终报告新增图表
├── prior_research/         # 旧报告快照
├── scripts/                # 一键组包与诊断生成脚本
└── SOURCE_MANIFEST.csv     # 源路径、快照方式、大小与 SHA256
```

## 重建

在项目根目录执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/conda/anaconda3/envs/base_93/bin/python \
research_delivery/scripts/build_tgd20_buyside_diagnostics.py

/opt/conda/anaconda3/envs/base_93/bin/python \
research_delivery/scripts/assemble_tgd20_research_package.py
```

## 数据边界

包内保存所有 TGD20 专属缓存与派生结果。DolphinDB 中的分钟行情、EOD、指数成分和行业库是共享原始数据库，不复制数据库本体；表名、字段、日期、复权和防未来函数口径记录在 `data/DATA_LINEAGE.md`。

大体积 parquet 在同一文件系统上优先使用 hard link。它们在本目录中是完整普通文件，移动或归档时可直接复制；这样避免在当前磁盘重复占用约 1 GB。
