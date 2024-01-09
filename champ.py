import streamlit as st
import pandas as pd
import numpy as np

from itertools import product
import pickle

from odds_helper import kelly, odds_to_prob, prob_to_odds

from name_helper import get_abbr

# def get_higher_seed(team1, team2):
#     return team1 if seeds[team1] < seeds[team2] else team2

df_spread_prob = pd.read_csv("spread_probs.csv")
ser_prob = df_spread_prob.set_index("Spread", drop=True)

def spread_to_prob(s):
    if isinstance(s, str):
        s = float(s)
    i = np.argmin(np.abs(ser_prob.index - s))
    return ser_prob.iloc[i].item()

rounds = ["WC", "DIV", "CONF", "SB"]

df = pd.read_csv("2023-wc.csv")
teams = df["Team"].values
df = df.set_index("Team")
team_seeds = df["Seed"]

pr_default = pd.read_csv("data/pr.csv")
pr_default = pr_default[pr_default["Team"].isin(teams)]
pr_default = pr_default.set_index("Team", drop=True)["PR"]
pr_default.loc["HFA"] = 1.3

# index: seeds
df = df.reset_index().set_index("Seed")

conf_dct = {conf:df[df["Conf"] == conf] for conf in df["Conf"].values}

st.set_page_config(page_title="Super Bowl markets")

st.title('Super Bowl markets')

with st.sidebar:

    st.header("Set your power ratings.")

    st.download_button(
        "Download a template", 
        data = pr_default.to_csv().encode('utf-8'),
        file_name = "power_ratings.csv",
        mime="text/csv"
        )

    pr_file = st.file_uploader("Accepts a csv file", type=["csv"])

    if pr_file:
        df_upload = pd.read_csv(pr_file)
        for name in ["Team", "team", "TEAM"]:
            if name in df_upload.columns:
                df_upload = df_upload.set_index("Team", drop=True)
                found = True
                break
        if not found:
            df_upload = df_upload.set_index(df_upload.columns[0], drop=True)
        if "PR" in df_upload.columns:
            pr_upload = df_upload["PR"]
        elif "pr" in df_upload.columns:
            pr_upload = df_upload["pr"]
        else:
            pr_upload = df_upload.iloc[:, 0]
        pr_upload2 = pd.Series()
        for k in pr_upload.index:
            if k.upper().strip() == "HFA":
                pr_upload2["HFA"] = pr_upload[k]
            abbr = get_abbr(k)
            if abbr:
                pr_upload2[abbr] = pr_upload[k]
        
        # Replace the uploaded ratings with the cleaned version
        pr_upload = pr_upload2

    pr = {}

    for t in teams:
        try:
            value = pr_upload[t]
        except (NameError, KeyError): 
            value = pr_default[t]
        pr[t] = st.slider(
            f'{t} power rating',
            -15.0, 15.0, float(value))
        
    st.subheader("Extra parameter:")
    try:
        value = pr_upload["HFA"]
    except (NameError, KeyError): 
        value = pr_default["HFA"]
    pr["HFA"] = st.slider(
        'Home field advantage',
        0.0, 10.0, float(value))
    
def get_pairs(teams):
    output = []
    while len(teams) > 0:
        output.append((teams[0], teams[-1]))
        teams = teams[1:-1]
    return output

def get_favorite(team1, team2):
    return team1 if team_seeds[team1] < team_seeds[team2] else team2

def get_prob(pair, outcome):
    winner = next(team for team in pair if team in outcome)
    loser = next(team for team in pair if team != winner)
    hfa = pr["HFA"] if winner == get_favorite(winner, loser) else -pr["HFA"]
    return spread_to_prob(pr[winner] - pr[loser] + hfa)
    
def wc_poss(df):
    bye_team = df.loc[1, "Team"]
    teams = list(df.loc[2:, "Team"])
    pairs = get_pairs(teams)
    outcomes = [[bye_team] + list(winners) for winners in product(*pairs)]
    outcomes = [sorted(teams, key=lambda team: team_seeds[team]) for teams in outcomes]
    probs = []
    for outcome in outcomes:
        prob = 1
        for pair in pairs:
            prob *= get_prob(pair, outcome)
        probs.append((prob, outcome))
    return probs

def div_poss(teams):
    pairs = get_pairs(teams)
    outcomes = [list(winners) for winners in product(*pairs)]
    outcomes = [sorted(teams, key=lambda team: team_seeds[team]) for teams in outcomes]
    probs = []
    for outcome in outcomes:
        prob = 1
        for pair in pairs:
            prob *= get_prob(pair, outcome)
        probs.append((prob, outcome))
    return probs

st.header("Possible outcomes")

# This is the probability of making the super bowl
# The sum of these probabilities is 2, not 1.
conf_champ_dct = {}

for conf in ["AFC", "NFC"]:
    probs = []
    wc_probs = wc_poss(conf_dct[conf])
    for p, outcome in wc_probs:
        for p2, outcome2 in div_poss(outcome):
            probs.append((p*p2, outcome2))
    
    champ_probs = []
    for p, conf_championship in probs:
        for outcome in conf_championship:
            prob = p*get_prob(conf_championship, outcome)
            champ_probs.append((prob, outcome))
    
    conf_prob_dct = {t:0 for t in conf_dct[conf]["Team"]}
    for p,t in champ_probs:
        conf_prob_dct[t] += p

    conf_champ_dct.update(conf_prob_dct)

matchups = list(product(conf_dct["AFC"]["Team"], conf_dct["NFC"]["Team"]))

prob_matchups = [(conf_champ_dct[t1]*conf_champ_dct[t2], (t1, t2)) for t1,t2 in matchups]

matchup_dct = {"-".join(m):p for p,m in prob_matchups}
# Want both orientations
temp = {"-".join(m[::-1]):p for p,m in prob_matchups}
matchup_dct.update(temp)

sb_probs = []
for p, matchup in prob_matchups:
    for outcome in matchup:
        prob = p*get_prob(matchup, outcome)
        sb_probs.append((prob, outcome))

sb_dct = {t:0 for t in teams}
for p,t in sb_probs:
    sb_dct[t] += p

def display_plus(s):
    if s[0] == "-":
        return s
    else:
        return "+"+s

def name_market(row):
    if row["market"] == "conference":
        return "Conference Champion"
    elif row["market"] == "super bowl":
        return "Super Bowl Champion"
    elif row["market"] == "exact matchup":
        return "Super Bowl Exact Matchup"
    
market = pd.read_csv("data/markets.csv")
market = market[market["odds"].notna()].copy()
prob_dct = {
    "conference": conf_champ_dct,
    "super bowl": sb_dct,
    "exact matchup": matchup_dct
}

kelly_list = []

for _, row in market.iterrows():
    proc = {}
    proc["Market"] = name_market(row)
    proc["Odds"] = row["odds"]
    market = row["market"]
    if market == "exact matchup":
        team = row["team1"]+"-"+row["team2"]
    else:
        team = row["team"]
    proc["Team"] = team
    proc["Prob"] = prob_dct[market][team]
    proc["Kelly"] = kelly(proc["Prob"], proc["Odds"])
    proc["Site"] = row["site"]
    kelly_list.append(proc)

df_kelly = pd.DataFrame(kelly_list)
rec = df_kelly[df_kelly["Kelly"] > 0].sort_values("Kelly", ascending=False)
rec["Odds"] = rec["Odds"].astype(str).map(display_plus)
rec = rec[["Team", "Market", "Odds", "Prob", "Site", "Kelly"]].reset_index(drop=True).copy()

st.write(rec)
