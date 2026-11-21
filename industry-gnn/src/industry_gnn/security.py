from __future__ import annotations

import getpass
import os


def get_tushare_token() -> str:
    """Read a token without printing or persisting it."""
    token = ""
    try:
        from google.colab import userdata  # type: ignore

        token = (userdata.get("TUSHARE_TOKEN") or "").strip()
    except Exception:
        token = (os.environ.get("TUSHARE_TOKEN") or "").strip()
    if not token:
        token = getpass.getpass("TuShare Token (hidden; kept only in memory): ").strip()
    if len(token) < 20:
        raise ValueError("A valid TuShare token was not provided.")
    return token


def create_tushare_client(token: str):
    import tushare as ts

    # Avoid ts.set_token(), which mutates global SDK state. The value remains local to this client.
    return ts.pro_api(token)

