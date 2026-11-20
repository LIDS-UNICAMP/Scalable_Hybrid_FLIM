"""OLS mínimo (numpy) com SE clássico e p-valor t — evita depender de statsmodels."""
import numpy as np
import pandas as pd
from scipy import stats


def ols(df, y, cats=(), nums=()):
    X = pd.get_dummies(df[list(cats)].astype(str), drop_first=True, dtype=float) if cats else pd.DataFrame(index=df.index)
    for n in nums:
        X[n] = df[n].astype(float)
    X.insert(0, "const", 1.0)
    Y = df[y].astype(float).values
    ok = np.isfinite(Y) & np.isfinite(X.values).all(1)
    Xv, Y = X.values[ok], Y[ok]
    beta, *_ = np.linalg.lstsq(Xv, Y, rcond=None)
    res = Y - Xv @ beta
    n, k = Xv.shape
    s2 = res @ res / max(n - k, 1)
    cov = s2 * np.linalg.pinv(Xv.T @ Xv)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), max(n - k, 1))
    return pd.DataFrame({"coef": beta, "se": se, "p": p}, index=X.columns), n
