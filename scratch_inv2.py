import os, json
os.environ["MARKET"] = "FOREX"
from src.config import get_settings
from src.core.candles import CandleStore
from sqlalchemy import text
s = get_settings(); sch=s.forex_db_schema
store = CandleStore(s.market_history_database_url or s.database_url, enable_timescale=s.market_history_enable_timescale, require_timescale=s.market_history_require_timescale, schema=sch)
with store.engine.connect() as c:
    # a FINAL decision payload (the trace)
    row=c.execute(text(f"select payload from \"{sch}\".decisions where final_action in ('PAPER_BUY','PAPER_SELL') order by created_at desc limit 1")).scalar()
    p=json.loads(row) if isinstance(row,str) else row
    print("FINAL decision payload top-keys:", sorted(p.keys()) if isinstance(p,dict) else type(p))
    for k in ('stage_history','risk','technical','mtf','ml','qwen','multi_timeframe','regime'):
        if isinstance(p,dict) and k in p: print(f"  has {k}:", (list(p[k].keys()) if isinstance(p[k],dict) else str(p[k])[:70]))
    # a position payload
    prow=c.execute(text(f'select payload from "{sch}".positions limit 1')).scalar()
    pp=json.loads(prow) if isinstance(prow,str) else prow
    print("\nposition payload keys:", sorted(pp.keys()) if isinstance(pp,dict) else pp)
    print("  status=", pp.get("status"), "has current_price=", "current_price" in pp, "has unrealized=", any('unreal' in k for k in pp))
