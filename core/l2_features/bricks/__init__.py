"""Shared L2 feature bricks.

Layering (Factor Factory):

  L2 raw (minute Active_*)
    → feature brick   (this package)
    → factor builder
    → neutralized factor
    → portfolio / backtest

Brick names describe the *observable*, not an economic story:
  - active_size       → active buy avg-size concentration (NOT \"institution\")
  - active_pressure   → amount-weighted (buy-sell)/(buy+sell)
  - (future) flow_density20, tgd20, ...
"""

from __future__ import annotations
