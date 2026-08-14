# TGD20 数据血缘与回测口径

## 原始数据库

| 层 | 来源 | 用途 |
|---|---|---|
| 分钟行情 | DolphinDB `Stock_one_minute`，连续竞价分钟 Close | 分钟收益、Gu、Gd、日内时段控制 |
| 日行情 | Wind A-share EOD tables | 复权 C2C 收益、Open/Close、成交额 |
| 衍生指标 | `dfs://WIND.ASHAREEODDERIVATIVEINDICATOR` | 流通市值 |
| 行业 | 中信历史行业面板 | 行业去均值与行业匹配组合 |
| 指数成分 | 历史动态成分函数 `get_stock_pool` | CSI300/500/1000 分股票池检验 |

原始数据库为项目共享数据底座，不在单因子包内复制。

## 包内缓存

- `data/cache/tgd_timing_daily/`：逐月 timing/residual 中间结果。
- `data/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet`：冻结 TGD20 宽表。
- `data/cache/tgd_panels/TGD20_long_20200101_20251231_w20.parquet`：含中间变量的长表。
- `data/analysis/`：最终买方诊断表与日报序列。

其余短窗口缓存是上述完整窗口的重复切片，不进入交付包。

## 时点与收益口径

1. 当日分钟数据生成 `TGD20_t`。
2. 回测强制 `signal_shift=1`，即 `TGD20_t` 只匹配下一交易日收益。
3. 正式回测收益调用 `Factor_Dev_Lib.get_Ret_Matrix(..., method="c2c")`，不得用未复权 Close 的简单百分比变化替代。
4. Headline 超额收益基准是同日、同一测试股票池、同时具备有效信号和有效收益的全部股票等权收益，不是十个分组收益的均值代理。
5. 正向因子选择 G10；多空诊断为 G10−G1。

## 样本

- 机制与稳定性缓存覆盖：2020-01-01 至 2025-12-31。
- 正式确认样本：2022-01-28 至 2025-12-31，约 951 个交易日。
- Discovery/confirmation 切分由框架固定，最终报告不在确认样本上重新调参。

## 已知数据限制

- EOD turnover 字段在当前 Wind 表不可用；历史 runner 的可投资性层使用 `amount / float_mktcap` 代理。组合实际换手来自持仓权重变化，不依赖该字段。
- 容量测算使用 ADV20 与参与率近似，不是逐笔冲击模型；结果只能视为容量筛查上限。
- 行业中性结果区分“信号做 size+industry 残差化”和“组合行业权重匹配”两种口径，报告不将二者混称。
