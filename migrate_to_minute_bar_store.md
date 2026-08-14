# 迁移指南：MinuteBarStore（统一分钟数据层）

## 背景

项目内曾并行存在多套分钟缓存：

| 旧路径 | 问题 |
|--------|------|
| `intraday.parquet` / `intraday_2024.parquet` / `intraday_2025.parquet` | 年文件巨大，列名为 DDB 原始大写 |
| `.tmp_intraday_*_months/` | 筛选用临时月文件，与正式缓存脱节 |
| `research/cache/smart_money_active_v2/minute_raw/minute_YYYYMM.parquet` | ActiveV2 专用；列已小写但缺 volume 等字段且会话过滤在入库时完成 |

**新真相源**：`research/cache/minute_bars/YYYYMM.parquet`  
入口模块：`minute_bar_store.py` → 类 `MinuteBarStore`

---

## 配置开关

`factor_config.py`：

```python
USE_MINUTE_BAR_STORE = True          # False 可回退旧 minute_raw 路径
MINUTE_BAR_CACHE_ROOT = Path("research/cache/minute_bars")
MINUTE_BAR_HISTORY_START = dt.datetime(2020, 1, 1)
```

环境变量覆盖：

- `MINUTE_BAR_CACHE_ROOT`
- `MINUTE_BAR_HISTORY_START`（或 `MINUTE_BAR_STORE_START`）

---

## 规范列名

| DDB 原始 | 规范输出 |
|----------|----------|
| Symbol | symbol |
| Date | date |
| Bartime | bartime（与 date 拼成完整时间戳） |
| Open/High/Low/Close/Volume/Amount/Adjfactor | open/high/low/close/volume/amount/adjfactor |
| Active_buy_volume / Active_sell_volume | active_buy_vol / active_sell_vol |
| Active_buy_amount / Active_sell_amount | active_buy_amt / active_sell_amt |
| Active_buy_count / Active_sell_count | active_buy_count / active_sell_count（兼容别名 `*_cnt`） |
| Bid/Ask_cancel_volume | bid_cancel_vol / ask_cancel_vol |
| Bid/Ask_cancel_count | bid_cancel_count / ask_cancel_count |

- **不在 Store 层复权**；上层继续用 `apply_minute_qc` 等处理。
- **默认不过滤连续竞价**；需要时调用 `apply_trading_hours(df)`（ActiveV2 兼容入口已自动过滤）。

---

## 基本用法

```python
from minute_bar_store import MinuteBarStore, apply_trading_hours, get_default_store

store = get_default_store()  # 读 factor_config / 环境变量

df = store.get_data("2024-05-01", "2024-05-31")
# 缺月自动从 DolphinDB 拉取并写入 research/cache/minute_bars/202405.parquet

# 增量：补最近 N 个自然日到当月缓存
store.update_recent(days=1)

# 会话过滤（对齐旧 ActiveV2）
df_sess = apply_trading_hours(df)
```

强制重拉某月：

```python
store.get_data("2024-05-01", "2024-05-31", force_reload=True)
```

---

## 下游迁移清单

### 已自动适配（开关打开时）

- `core/l2_features/smart_money_active_v2_builder.load_minute_active_raw`  
  → 当 `USE_MINUTE_BAR_STORE=True` 时走 `MinuteBarStore` + `apply_trading_hours`。  
  因此 APM / IdealAmplitude / IdealReversal / active_pressure bricks 等**间接受益**，无需立即改 import。

### 建议改为直接调用 Store

| 模块 | 现状 | 建议 |
|------|------|------|
| `run_l2_raw_feature_single_factor_screen.py` | `pull_intraday_from_ddb` / 年 parquet | `MinuteBarStore.get_data`；特征侧用规范列名或本地 rename |
| `core/l2_features/smart_money_panel_builder.py` | 自有 `minute_raw` | 改为 Store；保留 session 过滤 |
| 各类 smoke：`run_*_active_v2_smoke.py` | `load_minute_active_raw` | 可保持；或显式 `get_default_store()` |

扫描仍引用旧路径的文件：

```bash
/opt/conda/anaconda3/envs/base_93/bin/python -c \
  "from minute_bar_store import scan_legacy_usages; print('\\n'.join(scan_legacy_usages()))"
```

或：

```bash
/opt/conda/anaconda3/envs/base_93/bin/python minute_bar_store.py
```

---

## 旧文件处理策略（本阶段不做物理删除）

1. **不要删除** `intraday_2024/2025.parquet` 与旧 `minute_raw`，直到至少一次全链路回测通过 Store。
2. 可选软链（示例，按需执行）：

```bash
# 仅作提示示例 —— 确认 Store 月缓存齐全后再考虑
# ln -s research/cache/minute_bars research/cache/smart_money_active_v2/minute_raw_DEPRECATED_HINT
```

3. 磁盘回收：确认 `research/cache/minute_bars/` 已覆盖所需月份后，再人工归档/删除年文件与重复 `minute_raw`。

---

## 测试

```bash
cd /path/to/factor_dev
OMP_NUM_THREADS=1 /opt/conda/anaconda3/envs/base_93/bin/python -m unittest tests.test_minute_bar_store -v
```

测试使用 mock 拉取，**不依赖** DolphinDB。联调真实集群时可：

```python
from minute_bar_store import MinuteBarStore
with MinuteBarStore() as store:
    df = store.get_data("2024-05-01", "2024-05-05")
    store.print_legacy_hints()
```

---

## 回退

```python
# factor_config.py
USE_MINUTE_BAR_STORE = False
```

`load_minute_active_raw` 将重新读写 `research/cache/smart_money_active_v2/minute_raw/`。

---

## 与 P1 的关系

分钟择时热力图（扩展 `run_p2_intraday_heatmap.py`）**不依赖**本 Store，可并行开工。真分钟 Alpha 实现时建议直接 `MinuteBarStore.get_data` + 窄表输出接入 `Intraday_Factor_Test_Process`。
