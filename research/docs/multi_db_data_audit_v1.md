# Multi-DB Data Audit v1

Audited at (UTC): `2026-07-31T03:21:03.376073+00:00`

Scope: every connection constant in `COMMON_CONST.py`, using client
patterns from `DB_Demo.py`. Passwords are never written to artifacts.

## Endpoint status

| Endpoint | Engine | OK | L2-relevant | Role |
|----------|--------|----|-------------|------|
| `DATA_DB_CONN` | DolphinDB | yes | no | intraday minute bars (project default) |
| `DATA_DB_WIND` | Oracle | yes | yes | Wind EOD / reference |
| `DATA_DB_JUYUAN` | Oracle | yes | yes | 聚源 fundamentals |
| `DATA_DB_JUYUAN_2` | Oracle | yes | yes | 聚源 fundamentals (replica) |
| `DATA_DB_ZYYX2` | Oracle | yes | yes | 朝阳永续 consensus |
| `DATA_DB_ORCL` | Oracle | yes | yes | generic ORCL |
| `DATA_DB_CAIHUI` | Oracle | yes | yes | 财汇 |
| `DATA_DB_CAIHUI_2` | Oracle | yes | yes | 财汇 replica |
| `DATA_DB_PAIPAI` | Oracle | yes | yes | 排排网 |
| `DATA_DB_PAIPAI_2` | Oracle | yes | yes | 排排网 replica |
| `DATA_DB_PUYI` | Oracle | yes | yes | 普益 |
| `DATA_DB_PUYI_2` | Oracle | yes | yes | 普益 replica |
| `DATA_DB_DATAYES` | MySQL | yes | no | 通联 datayes |
| `DATA_DB_YECHEN` | MySQL | yes | no | 野尘 |
| `DATA_DB_YECHEN_2` | MySQL | yes | no | 野尘 replica |
| `DATA_DB_EMDATA` | MySQL | yes | no | 东方财富 |
| `DATA_DB_EMDATA_2` | MySQL | yes | no | 东方财富 replica |
| `DATA_DB_HFDATA` | ClickHouse | yes | yes | 高频行情 / L2 candidate (cmds) |

## ClickHouse HF findings

- Databases visible: cmds
- Table counts: `{"cmds": 44}`
- Demo `CFFEX_AL_KLIN_RTH` ok=True
- Snapshot-like tables (24):
  - `cmds.CFFEX_AL_SSL1_RTH`
  - `cmds.CFFEX_AL_SSL2_EXG`
  - `cmds.CUSE_AL_SSL1_RTH`
  - `cmds.CZCE_AL_SSL1_RTH`
  - `cmds.CZCE_AL_SSL2_EXG`
  - `cmds.DCE_AL_SSL1_RTH`
  - `cmds.DCE_AL_SSL2_EXG`
  - `cmds.HKEX_EQ_SSL1_RTH`
  - `cmds.HKEX_EQ_SSL2_RTH`
  - `cmds.LOCAL_CFFEX_AL_SSL1_RTH`
  - `cmds.LOCAL_CFFEX_AL_SSL2_EXG`
  - `cmds.LOCAL_CUSE_AL_SSL1_RTH`
  - `cmds.LOCAL_CZCE_AL_SSL1_RTH`
  - `cmds.LOCAL_CZCE_AL_SSL2_EXG`
  - `cmds.LOCAL_DCE_AL_SSL1_RTH`
  - `cmds.LOCAL_DCE_AL_SSL2_EXG`
  - `cmds.LOCAL_HKEX_EQ_SSL1_RTH`
  - `cmds.LOCAL_HKEX_EQ_SSL2_RTH`
  - `cmds.LOCAL_SHFE_AL_SSL1_RTH`
  - `cmds.LOCAL_SSE_AL_SSL2_EXG`
  - `cmds.LOCAL_SZSE_AL_SSL2_EXG`
  - `cmds.SHFE_AL_SSL1_RTH`
  - `cmds.SSE_AL_SSL2_EXG`
  - `cmds.SZSE_AL_SSL2_EXG`
- Tick-like tables (4):
  - `cmds.LOCAL_SSE_AL_TICK_EXG`
  - `cmds.LOCAL_SZSE_AL_TICK_EXG`
  - `cmds.SSE_AL_TICK_EXG`
  - `cmds.SZSE_AL_TICK_EXG`

### Top scored HF tables

- `cmds.CZCE_AL_SSL2_EXG` cols=41 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['ToltalBidVolume', 'ToltalAskVolume', 'ArbBidPrice', 'ArbAskPrice', 'ArbBidVolume', 'ArbAskVolume', 'BidPrices', 'AskPrices', 'BidVolumes', 'AskVolumes']
- `cmds.LOCAL_CZCE_AL_SSL2_EXG` cols=41 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['ToltalBidVolume', 'ToltalAskVolume', 'ArbBidPrice', 'ArbAskPrice', 'ArbBidVolume', 'ArbAskVolume', 'BidPrices', 'AskPrices', 'BidVolumes', 'AskVolumes']
- `cmds.LOCAL_SSE_AL_SSL2_EXG` cols=54 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes', 'TotalBidVolume', 'TotalAskVolume', 'BidPriceNum', 'AskPriceNum']
- `cmds.SSE_AL_SSL2_EXG` cols=54 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes', 'TotalBidVolume', 'TotalAskVolume', 'BidPriceNum', 'AskPriceNum']
- `cmds.DCE_AL_SSL2_EXG` cols=35 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'ArbBidVolumes', 'AskPrices', 'AskVolumes', 'ArbAskVolumes']
- `cmds.LOCAL_DCE_AL_SSL2_EXG` cols=35 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'ArbBidVolumes', 'AskPrices', 'AskVolumes', 'ArbAskVolumes']
- `cmds.LOCAL_SZSE_AL_SSL2_EXG` cols=53 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['TotalAskVolume', 'TotalBidVolume', 'BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.SZSE_AL_SSL2_EXG` cols=53 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['TotalAskVolume', 'TotalBidVolume', 'BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.CFFEX_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.CFFEX_AL_SSL2_EXG` cols=27 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.CUSE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.CZCE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.DCE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.HKEX_EQ_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.HKEX_EQ_SSL2_RTH` cols=10 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.LOCAL_CFFEX_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_CFFEX_AL_SSL2_EXG` cols=27 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.LOCAL_CUSE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_CZCE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_DCE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_HKEX_EQ_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_HKEX_EQ_SSL2_RTH` cols=10 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrices', 'BidVolumes', 'AskPrices', 'AskVolumes']
- `cmds.LOCAL_SHFE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.SHFE_AL_SSL1_RTH` cols=25 bid=True ask/offer=True trade=False orderid=False sample_n=2 hits=['BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
- `cmds.LOCAL_SZSE_AL_TICK_EXG` cols=37 bid=False ask/offer=False trade=False orderid=True sample_n=2 hits=['SecondaryOrderID']

## A-share native L2 (the real unlock)

ClickHouse `DATA_DB_HFDATA` / database `cmds` **does** carry equity snapshot and
tick. Primary research tables:

| Table | Role | Key fields |
|-------|------|------------|
| `SSE_AL_SSL2_EXG` | 上交所 L2 snapshot | `BidPrices/Volumes/Nums`, `AskPrices/Volumes/Nums` (Array), withdraw stats, `BidVWAP`/`AskVWAP` |
| `SZSE_AL_SSL2_EXG` | 深交所 L2 snapshot | same LOB arrays + match fields |
| `SSE_AL_TICK_EXG` | 上交所 tick | `Price/Volume/Amount`, `BidOrderNo`/`AskOrderNo`, `BSFlag` |
| `SZSE_AL_TICK_EXG` | 深交所 tick | tick + richer order metadata |
| `SSE_AL_KLIN_EXG` / `SZSE_AL_KLIN_CMD` | HF bars | OHLCVA (parallel to DDB minute path) |

`LOCAL_*` mirrors of the same tables also exist. Futures/commodity SSL1/SSL2
tables are present but out of scope for the current A-share ZZ1000 library.

Array-vector depth (`BidPrices` etc.) is exactly the input needed for
microprice / weighted OI / depth imbalance — previously marked blocked under
DolphinDB-only Phase 0.

## Interpretation for Sprint 4.4

- **Corrected gate:** native L2 is available via ClickHouse, not DolphinDB.
- DolphinDB `DATA_DB_CONN` remains the production minute-bar / evaluation path.
- Sprint 4.4 Phase 1 should build a **ClickHouse → (optional DDB aggregate) →
  `intraday_evaluation_v2`** bridge, not a DDB `QV_Snapshot` loader.
- Oracle endpoints marked `l2_relevant=yes` are **name-pattern false positives**
  (`%L2%` / `%TICK%` in unrelated object names). They are not equity LOB feeds.
- MySQL endpoints have no LOB/tick schema under this audit.

JSON: `research/results/multi_db_data_audit.json`
