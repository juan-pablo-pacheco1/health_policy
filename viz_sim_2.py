#!/usr/bin/env python3
# viz_multi_scenarios.py

import os
import webbrowser
import pandas as pd
import numpy as np
import copy
import plotly.express as px

# ─── 1) DATA CLEAN & PREP ────────────────────────────────────────────────────

# map codes for child race → human labels:
RACE_MAP = {
    '1': "American Indian/Alaska Native",
    '2': "Black/African American",
    '3': "Asian",
    '4': "Native Hawaiian/Other Pacific Islander",
    '5': "White",
    '6': "Hispanic/Latino",
    '7': "Multiracial",
    '8': "Some other race"
}

def load_and_clean(df_raw):
    wellbeing = ['lonely_freq_s2','fs_feeling_s2','fs_worry_s2']
    childcol  = ['missed_days_s2']
    df = df_raw.copy()
    # numeric coercion
    for c in wellbeing + childcol:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop missing/invalid
    df = df.dropna(subset=wellbeing + childcol)
    df = df[df[wellbeing].isin([1,2,3,4,5]).all(axis=1)]
    return df

def make_families(df):
    fams = []
    for _, row in df.iterrows():
        code = str(row.get('race_ethn_cmc','')).strip()
        race = RACE_MAP.get(code, "Prefer not/missing")
        fams.append({
            'id':         row['record_id'],
            'baseline':   row['missed_days_s2'],
            'overwhelm':  row['fs_feeling_s2'],
            'worry':      row['fs_worry_s2'],
            'loneliness': row['lonely_freq_s2'],
            'race':       race
        })
    return fams

# ─── 2) SIMULATION CORE ─────────────────────────────────────────────────────

def simulate_family(fam, betas, policy, seed=None):
    if seed is not None:
        np.random.seed(seed)
    base  = fam['baseline']
    # apply policy delta directly (betas may be positive/negative)
    saved = sum(betas[m] * policy.get(m, 0) for m in betas)
    expct = max(0, base - saved)
    f2    = copy.deepcopy(fam)
    f2['after'] = np.random.poisson(lam=expct)
    return f2

def run_sim(fams, betas, policy):
    return [simulate_family(f, betas, policy) for f in fams]

# ─── 3) BUILD ANIMATION DATAFRAME ───────────────────────────────────────────

def build_anim_df(fams, betas, policy, steps=30):
    base_run = run_sim(fams, betas, {m:0 for m in betas})
    pol_run  = run_sim(fams, betas, policy)
    records = []
    emojis = ['👦','👧','🧑','🧓','👩']
    for f0, f1 in zip(base_run, pol_run):
        e = np.random.choice(emojis)
        for t in np.linspace(0,1,steps):
            y = f0['baseline'] + (f1['after'] - f0['baseline']) * t
            records.append({
                'family':  f0['id'],
                'emoji':   e,
                'y':        y,
                'frame':    round(t,3),
                'race':     f0['race']
            })
    return pd.DataFrame(records)

# ─── 4) MAIN: LOOP SCENARIOS, WRITE FILES & BUILD INDEX ────────────────────

if __name__=="__main__":

    # your regression‑derived betas
    BETAS = {
      'overwhelm':  -0.0379915,
      'worry':      -0.1098821,
      'loneliness':  0.453555
    }

    # read only needed cols as strings
    df_raw = pd.read_csv("data.csv", dtype=str)
    keep = [
      'record_id','race_ethn_cmc',
      'lonely_freq_s2','fs_feeling_s2','fs_worry_s2','missed_days_s2'
    ]
    df_clean = load_and_clean(df_raw[keep])
    families = make_families(df_clean)

    # define scenarios (drop no‑policy)
    scenarios = {
      'ads_simple':     {'loneliness':  0.0},
      'cbt':            {'loneliness': -0.27},
      'animal_therapy': {'loneliness': -0.80},
    }

    out_files = []
    all_anim   = []

    # generate each scenario’s animation + collect for time‑series
    for name, policy in scenarios.items():
        anim_df = build_anim_df(families, BETAS, policy, steps=30)
        pretty = name.replace('_',' ').title()
        anim_df['scenario'] = pretty
        all_anim.append(anim_df)

        title = f"🏠 Family Absence Game: Before → After Policy ({pretty})"
        fig = px.scatter(
            anim_df,
            x="family", y="y",
            animation_frame="frame",
            text="emoji",
            range_y=[-0.5, anim_df['y'].max()+1],
            title=title,
            hover_data=['family','emoji','y','frame','race']
        )
        fig.update_traces(
            textposition="middle center",
            marker=dict(size=1, opacity=0)
        )
        fig.update_layout(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(title="Absence Days"),
            font=dict(size=18)
        )
        slider = fig.layout.sliders[0]
        slider.currentvalue.prefix = "Progress: "
        slider.currentvalue.font.size = 14

        outfn = f"viz_{name}.html"
        fig.write_html(outfn, include_plotlyjs='cdn')
        print(f"→ wrote {outfn}")
        out_files.append((pretty, outfn))

    # ─── 5) OVER‑TIME INTERACTIVE COMPARISON ─────────────────────────────────

    # combine all animation records
    time_df = pd.concat(all_anim, ignore_index=True)
    # compute average absence at each frame by scenario
    time_ts = (
        time_df
        .groupby(['scenario','frame'])['y']
        .mean()
        .reset_index(name='avg_absences')
    )

    fig2 = px.line(
        time_ts,
        x="frame", y="avg_absences",
        color="scenario",
        title="🏠 Average Absences Over Progress (by Scenario)",
        labels={"frame":"Progress","avg_absences":"Avg Absence Days"}
    )
    fig2.update_layout(
        hovermode="x unified",
        legend_title="Scenario",
        font=dict(size=18)
    )

    time_fn = "viz_compare_time.html"
    fig2.write_html(time_fn, include_plotlyjs='cdn')
    print(f"→ wrote {time_fn}")
    out_files.append(("Over‑Time Comparison", time_fn))

    # ─── 6) BUILD AND OPEN INDEX ───────────────────────────────────────────────

    idx = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Family Absence Scenarios</title></head>",
        "<body style='font-family:sans-serif; margin:2rem;'>",
        "<h1>📊 Family Absence Scenarios</h1>",
        "<ul>"
    ]
    for label, fn in out_files:
        idx.append(f"  <li><a href='{fn}' target='_blank'>{label}</a></li>")
    idx += ["</ul></body></html>"]

    with open("index.html","w") as f:
        f.write("\n".join(idx))
    print("→ wrote index.html")

    # attempt to open in browser
    webbrowser.open("file://" + os.path.abspath("index.html"))
    print("Done! Opening index.html…")
