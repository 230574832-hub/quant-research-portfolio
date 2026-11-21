"""Build the A-share research panel (2011-2026) from the local sqlite + support CSVs.

Output: data/panel_wide.pkl  dict of wide float32 matrices (date x code) + metadata.
"""
import pickle
import sqlite3
import numpy as np
import pandas as pd

DB = "data/daily_panel_v12.sqlite"
END = "2026-03-25"          # after this date ret/adj_close/mktcap are missing
OUT = "data/panel_wide.pkl"

print("loading sqlite ...")
con = sqlite3.connect(DB)
df = pd.read_sql_query(
    f"""select code, date, open, high, low, close, preclose, adj_close,
               volume, amount, mktcap, mktcap2, ret, limit_status
        from daily where date <= '{END}' and ret is not null""",
    con,
)
con.close()
print("rows:", len(df), "codes:", df.code.nunique())

df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates(["code", "date"]).sort_values(["date", "code"])
print("after dedupe:", len(df))

# ---- board from code prefix ------------------------------------------------
def board_of(c):
    if c.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "main"
    if c.startswith(("300", "301")):
        return "chinext"
    if c.startswith(("688", "689")):
        return "star"
    if c.startswith(("8", "4", "920")):
        return "bse"
    return "other"

code_board = pd.Series(df.drop_duplicates("code")["code"].map(board_of).values,
                       index=df.drop_duplicates("code")["code"].values)

# ---- company info: industry, listing date ----------------------------------
ci = pd.read_csv("data/support/company_info.csv", dtype=str)
ci = ci[ci.Scode.str.match(r"^\d{6}$", na=False)]
ci = ci.set_index("Scode")
industry = ci["IndusB"].str.strip()
listdt = pd.to_datetime(ci["Listdt"], errors="coerce")

# ---- ST history -> daily state ---------------------------------------------
st = pd.read_csv("data/support/st_history.csv", dtype=str)
st = st[st.SCode.str.match(r"^\d{6}$", na=False)].copy()
st["ImpDate"] = pd.to_datetime(st["ImpDate"], errors="coerce")
st = st.dropna(subset=["ImpDate"]).sort_values("ImpDate")


def build_st_matrix(dates, codes):
    """Event-sourced ST flag, forward-filled across trading dates."""
    mat = np.zeros((len(dates), len(codes)), dtype=bool)
    idx = {c: i for i, c in enumerate(codes)}
    didx = {d: i for i, d in enumerate(dates)}
    for code, g in st.groupby("SCode"):
        if code not in idx:
            continue
        c = idx[code]
        events = []
        for imp, tcat in zip(g["ImpDate"], g["TCat"].fillna("")):
            if imp not in didx:
                continue
            state = None
            if tcat in ("被ST", "被*ST", "ST变为*ST", "*ST变为ST", "ST变为PT", "PT变为ST"):
                state = True
            elif tcat in ("撤销ST", "撤销*ST"):
                state = False
            elif tcat in ("退市", "撤销*ST（退市整理期）"):
                state = True
            if state is not None:
                events.append((didx[imp], state))
        if not events:
            continue
        events.sort()
        cur = False
        ptr = 0
        for i in range(len(dates)):
            while ptr < len(events) and events[ptr][0] <= i:
                cur = events[ptr][1]
                ptr += 1
            if cur:
                mat[i, c] = True
    return mat


# ---- share structure: annual snapshots, forward fill ------------------------
ss = pd.read_csv("data/support/share_structure.csv", dtype=str)
ss = ss[ss.SCode.str.match(r"^\d{6}$", na=False)].copy()
ss["Date"] = pd.to_datetime(ss["Date"], errors="coerce")
for col in ("TotlShr", "CirShr_A"):
    ss[col] = pd.to_numeric(ss[col], errors="coerce")
ss = ss.dropna(subset=["Date", "CirShr_A"]).sort_values("Date")

print("pivoting ...")
w = {}
for col, name in [("open", "open"), ("high", "high"), ("low", "low"),
                  ("close", "close"), ("preclose", "preclose"),
                  ("adj_close", "adjclose"), ("volume", "volume"),
                  ("amount", "amount"), ("mktcap", "mktcap"),
                  ("ret", "ret")]:
    w[name] = df.pivot(index="date", columns="code", values=col).astype(np.float32)

dates = w["close"].index
codes = w["close"].columns

# sanity: ret vs adj_close returns
adj_ret = w["adjclose"].pct_change(fill_method=None)
print("corr(ret, adjclose pct_change) =",
      pd.Series(w["ret"].stack()).corr(pd.Series(adj_ret.stack())).round(4))

st_mat = build_st_matrix(dates, codes)

# shares -> daily matrix via forward fill on calendar
cir = ss.pivot_table(index="Date", columns="SCode", values="CirShr_A", aggfunc="last")
cir = cir.reindex(cir.index.union(dates)).sort_index().ffill().reindex(dates)
cir = cir.reindex(columns=codes).astype(np.float32)

board = code_board.reindex(codes).fillna("other")
ind = industry.reindex(codes).fillna("未知")
ldt = listdt.reindex(codes)

# listed days since first appearance in panel
first_seen = w["close"].notna().idxmax()

panel = {
    "open": w["open"], "high": w["high"], "low": w["low"], "close": w["close"],
    "preclose": w["preclose"], "adjclose": w["adjclose"],
    "volume": w["volume"], "amount": w["amount"], "mktcap": w["mktcap"],
    "ret": w["ret"], "cir_shr": cir, "is_st": st_mat,
    "dates": dates, "codes": codes,
    "board": board, "industry": ind, "list_date": ldt, "first_seen": first_seen,
}
pickle.dump(panel, open(OUT, "wb"), protocol=4)
print("saved", OUT)
print("shape:", w["close"].shape, "| ST rows:", int(st_mat.sum()))
print("board counts:", board.value_counts().to_dict())
print("industry n:", ind.nunique())
