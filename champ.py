import streamlit as st
import pandas as pd
import numpy as np

from itertools import product
import pickle

from odds_helper import odds_to_prob, prob_to_odds

abbr_dict = {
    "Chiefs": "KC",
    "Bengals": "CIN",
    "Eagles": "PHI",
    "49ers": "SF",
}

teams = abbr_dict.values()

def get_abbr(s):
    return abbr_dict[s.split()[-1]]

st.title('Super Bowl markets')


st.header("Projected spreads")

st.markdown(
"""Enter the spread, with a **positive number meaning the listed team is favored**. For example, if you enter 2.5 for KC vs PHI, that means you project Kansas City to be favored by 2.5, and if you enter -2.5, then you project PHI to be favored by 2.5.""")

def make_matchup(row):
    return st.text_input(f"{row['Team1']} vs {row['Team2']}.  Projection: {row['Favorite']} by")

def display_matchups(week, df_spreads, spread_dict):
    df_sub = df_spreads[df_spreads["Week"] == week]
    for _, row in df_sub.iterrows():
        res = make_matchup(row)
        spread_dict[(row["Team1"], row["Team2"])] = res

df_spreads = pd.read_csv("spreads-Jan22.csv")

conf_rd = df_spreads[df_spreads["Week"] == 2]
sb_rd = df_spreads[df_spreads["Week"] == 3]

teams = list(set([t for t in df_spreads["Team1"]] +[t for t in df_spreads["Team2"]]))

def get_favorite(team1, team2):
    return df_spreads[df_spreads["Team1"].isin([team1, team2]) &
        df_spreads["Team2"].isin([team1, team2])]["Favorite"].item()

spread_dict = {}

col1, _, col2, _2 = st.columns([5,2, 5, 3])

with col1:
    st.subheader("Conference championship round")
    display_matchups(2, df_spreads, spread_dict)

with col2:
    st.subheader("Super Bowl")
    display_matchups(3, df_spreads, spread_dict)

def get_spread(team1, team2):
    try:
        return spread_dict[(team1, team2)]
    except KeyError:
        return spread_dict[(team2, team1)]

df_spread_prob = pd.read_csv("spread_probs.csv")
ser_prob = df_spread_prob.set_index("Spread", drop=True)

def spread_to_prob(s):
    if isinstance(s, str):
        s = float(s)
    i = np.argmin(np.abs(ser_prob.index - s))
    return ser_prob.iloc[i].item()

# b bool for whether favorite wins
def process_row(row, b):
    team1, team2 = row[["Team1", "Team2"]]
    fav = get_favorite(team1, team2)
    und = team1 if fav == team2 else team2
    fav_spread = get_spread(team1, team2)
    winner = fav if b else und
    try:
        fav_spread = float(fav_spread)
        spread = fav_spread if b else -fav_spread
        prob = spread_to_prob(spread)
    except:
        raise ValueError(f"Check the spread for {team1} vs {team2}")
    return {"winner": winner, "prob": prob}


def process_rd(df, tup):
    if len(df) != len(tup):
        raise ValueError("Wrong df and tup")
    return pd.DataFrame(
            [process_row(df.loc[i], b) for i, b in zip(df.index, tup)]
        )


def run_sim(tup):
    conf_matchups = conf_rd
    conf_outcome = process_rd(conf_matchups, tup[:2])
    conf_winners = conf_outcome["winner"].values
    sb_matchup = sb_rd[sb_rd["Team1"].isin(conf_winners) & sb_rd["Team2"].isin(conf_winners)]
    sb_outcome = process_rd(sb_matchup, tup[2:])
    df_outcome = pd.concat([conf_outcome, sb_outcome], axis=0).reset_index(drop=True)
    prob = np.prod(df_outcome["prob"])
    return (df_outcome["winner"], prob)

sb_name = "SUPER BOWL - Odds to Win"
fin_name = "SUPER BOWL - Exact Finalists"
res_name = "SUPER BOWL - Exact Result"
lose_name = "SUPER BOWL - Losing Team"

markets = [sb_name, fin_name, res_name, lose_name]

results = {}
for name in markets:
    results[name] = {}


def replace(match, sep):
    return f" {sep} ".join([get_abbr(x) for x in match.split(f" {sep} ")])


def update_prob(dct, k, p):
    dct[k] = dct.get(k,0) + p


try:
    for outcome in product([True, False], repeat=3):
        ser_outcome, p = run_sim(outcome)
        # st.write(ser_outcome)
        sb_winner = ser_outcome.iloc[-1]
        update_prob(results[sb_name], sb_winner, p)
        # AFC team always written first
        afc_champ = ser_outcome.iloc[-3]
        nfc_champ = ser_outcome.iloc[-2]
        sb_loser = next(t for t in (afc_champ, nfc_champ) if t != sb_winner)
        update_prob(results[fin_name], f"{afc_champ} vs {nfc_champ}", p)
        update_prob(results[res_name], f"{sb_winner} to beat {sb_loser}", p)
        update_prob(results[lose_name], sb_loser, p)
except ValueError:
    st.write("Enter all spreads above to see the results")
    

def display_results(name):
    st.header(name)

    probs = results[name]
    sorted_keys = sorted(probs.keys(), key=lambda k: probs[k], reverse=True)

    for k in sorted_keys:
        st.markdown(f"**{k}**. Computed fair odds {prob_to_odds(probs[k])}")

for name in markets:
    display_results(name)


# st.header("AFC Championship")

# df = pd.read_csv("2023-div-rd.csv")
# df = df.set_index("Team")
# seeds = df["Seed"]



# def get_higher_seed(team1, team2):
#     return team1 if seeds[team1] < seeds[team2] else team2

# placeholder = st.empty()

# use_prob = st.checkbox('Use probabilities instead of spreads')

# if use_prob:
#     placeholder.markdown(
# """Enter the probability (as a decimal between 0 and 1) of the higher seed winning. For example, if you enter 0.8 for KC vs BUF, that means 80% chance KC wins.""")
# else:
#     placeholder.markdown(
# """Enter the spread, with a **positive number meaning the higher seed is favored**. For example, if you enter -2 for KC vs BUF, that means you project Buffalo to be favored by 2, and if you enter 2.5, then you project KC to be favored by 2.5.""")

# col1, col2, buff = st.columns([5,5,5])

# def make_matchup(team1, team2):
#     hteam = get_higher_seed(team1, team2)
#     ateam = team1 if hteam == team2 else team2
#     return st.text_input(f"{hteam} vs {ateam}")

# div_pairs = [("KC", "JAX"), ("BUF", "CIN")]
# teams = [team for pair in div_pairs for team in pair]

# conf_pairs = list(product(*div_pairs))

# val_dict = {}

# with col1:
#     st.write(f"Enter your predicted {'probability' if use_prob else 'spread'}.")
#     st.subheader("Divisional round")
#     for pair in div_pairs:
#         res = make_matchup(*pair)
#         val_dict[pair] = res
#     if res != "":
#         hteam = get_higher_seed(*pair)
#         ateam = pair[0] if hteam == pair[1] else pair[1]
#         p = float(res) if use_prob else spread_to_prob(res)
#         st.write(f"Example: We estimate {hteam} has a {p:.0%} chance of beating {ateam} in the divisional round.")

# with col2:
#     st.subheader("Conference championship")
#     for pair in conf_pairs:
#         val_dict[pair] = make_matchup(*pair)



# def get_team_prob(team, prob_dict):
#     orig = next(pair for pair in div_pairs if team in pair)
#     if get_higher_seed(*orig) == team:
#         prob = prob_dict[orig]
#     else:
#         prob = 1 - prob_dict[orig]
#     opp_pair = next(pair for pair in div_pairs if team not in pair)
    
#     next_prob = 0
#     for pair in conf_pairs:
#         if team not in pair:
#             continue
#         if team == get_higher_seed(*pair):
#             temp_prob = prob_dict[pair]
#         else:
#             temp_prob = 1-prob_dict[pair]
#         opp = next(t for t in pair if t != team)
#         if opp == get_higher_seed(*opp_pair):
#             opp_prob = prob_dict[opp_pair]
#         else:
#             opp_prob = 1-prob_dict[opp_pair]
#         next_prob += opp_prob*temp_prob
    
#     return prob*next_prob


# st.subheader("Estimated fair prices to be AFC Champion:")

# if (len(val_dict) == 6) and all(v != "" for v in val_dict.values()):
#     if use_prob:
#         prob_dict = {k:float(v) for k,v in val_dict.items()}
#     else:
#         prob_dict = {k:spread_to_prob(v) for k,v in val_dict.items()}
#     for team in teams:
#         prob = get_team_prob(team, prob_dict)
#         st.write(f"{team}: {prob_to_odds(prob)} (probability: {prob:.3f})")
# else:
#     st.write("(Be sure to enter all six values.)")
    
# st.header("NFC Championship")

# placeholder = st.empty()

# use_prob_nfc = st.checkbox('Use probabilities instead of spreads', key="prob_checkbox")

# col1, col2, buff = st.columns([5,5,5])

# def make_matchup(team1, team2):
#     hteam = get_higher_seed(team1, team2)
#     ateam = team1 if hteam == team2 else team2
#     return st.text_input(f"{hteam} vs {ateam}")

# wc_pair = ("TB", "DAL")
# div_pairs = [("PHI", "NYG"), ("SF", "TB"), ("SF", "DAL")]
# teams = list({team for pair in div_pairs for team in pair})

# conf_pairs = list(product(("PHI", "NYG"), ("SF", "TB", "DAL")))

# val_dict = {}

# with col1:
#     st.write(f"Enter your predicted {'probability' if use_prob_nfc else 'spread'}.")
#     st.subheader("Wild-card round")
#     res = make_matchup(*wc_pair)
#     val_dict[wc_pair] = res
#     st.subheader("Divisional round")
#     for pair in div_pairs:
#         res = make_matchup(*pair)
#         val_dict[pair] = res
#     if res != "":
#         hteam = get_higher_seed(*pair)
#         ateam = pair[0] if hteam == pair[1] else pair[1]
#         p = float(res) if use_prob_nfc else spread_to_prob(res)
#         st.write(f"Example: We estimate {hteam} has a {p:.0%} chance of beating {ateam} in the divisional round.")

# with col2:
#     st.subheader("Conference championship")
#     for pair in conf_pairs:
#         val_dict[pair] = make_matchup(*pair)


# def get_team_prob(team, prob_dict):
#     output_prob = 0

#     # Assume wc_team wins the wild card round
#     for wc_team in wc_pair:
#         if (team in wc_pair) and (team != wc_team):
#             continue
#         scale = get_pair_prob(wc_team, wc_pair)
#         hyp_div = [("PHI", "NYG"), ("SF", wc_team)]
#         hyp_teams = [t for pair in hyp_div for t in pair]

#         orig = next(pair for pair in hyp_div if team in pair)
#         prob = get_pair_prob(team, orig)
#         opp_pair = next(pair for pair in hyp_div if team not in pair)

#         conf_pairs = product(*hyp_div)
        
#         next_prob = 0
#         for pair in conf_pairs:
#             if team not in pair:
#                 continue
#             temp_prob = get_pair_prob(team, pair)
#             opp = next(t for t in pair if t != team)
#             opp_prob = get_pair_prob(opp, opp_pair)
#             next_prob += opp_prob*temp_prob
        
#         output_prob += scale*prob*next_prob
    
#     return output_prob

# def get_pair_prob(team, pair):
#     p = prob_dict[pair]
#     if get_higher_seed(*pair) == team:
#         return p
#     else:
#         return 1-p

# st.subheader("Estimated fair prices to be NFC Champion:")

# `all_probs = {}

# if (len(val_dict) == 10) and all(v != "" for v in val_dict.values()):
#     if use_prob_nfc:
#         prob_dict = {k:float(v) for k,v in val_dict.items()}
#     else:
#         prob_dict = {k:spread_to_prob(v) for k,v in val_dict.items()}

#     for team in teams:
#         prob = get_team_prob(team, prob_dict)
#         all_probs[team] = prob

#     sorted_keys = sorted(all_probs.keys(), key=lambda k: all_probs[k], reverse=True)
#     for team in sorted_keys:
#         prob = all_probs[team]
#         st.write(f"{team}: {prob_to_odds(prob)} (probability: {prob:.3f})")
# else:
#     st.write("(Be sure to enter all ten values.)")
