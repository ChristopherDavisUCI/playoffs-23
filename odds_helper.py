import numpy as np
import pandas as pd

def prob_to_odds(p):
    if p < .000001:
        return np.nan
    if p > .999999:
        return np.nan
    if p > 0.5:
        x = 100*p/(p-1)
        return f"{x:.0f}"
    elif p <= 0.5:
        x = 100*(1-p)/p
        return f"+{x:.0f}"

def odds_to_prob(x):
    x = float(x)
    if x < 0:
        y = -x
        return y/(100+y)
    else:
        return 100/(100+x)
