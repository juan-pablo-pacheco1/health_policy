#!/usr/bin/env python3
# viz_multi_scenarios.py

import os
import pandas as pd
import numpy as np
import copy
import plotly.express as px

# ─── 1) DATA CLEAN & PREP ────────────────────────────────────────────────────

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
    well  = ['lonely_freq_s2','fs_feeling_s2','fs_worry_s2']
    child = ['missed_days_s2']
    df = df_raw.copy()
    # numeric coercion
    for c in well + child:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop missing/invalid
    df = df.dropna(subset=well + child)
    df = df[df[well].isin([1,2,3,4,5]).all(axis=1)]
    return df

def make_families(df):
    fams = []
    for _, r in df.iterrows():
        code = str(r.get('race_ethn_cmc','')).strip()
        race = RACE_MAP.get(code, "Prefer not/missing")
        fams.append({
            'id':         r['record_id'],
            'baseline':   r['missed_days_s2'],
            'overwhelm':  r['fs_feeling_s2'],
            'worry':      r['fs_worry_s2'],
            'loneliness': r['lonely_freq_s2'],
            'race':       race
        })
    return fams

# ─── 2) SIMULATION CORE ─────────────────────────────────────────────────────

def simulate_family(fam, betas, policy, seed=None):
    if seed is not None:
        np.random.seed(seed)
    base  = fam['baseline']
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
    recs = []
    emojis = ['👦','👧','🧑','🧓','👩']
    for f0, f1 in zip(base_run, pol_run):
        e = np.random.choice(emojis)
        for t in np.linspace(0,1,steps):
            y = f0['baseline'] + (f1['after'] - f0['baseline']) * t
            recs.append({
                'family': f0['id'],
                'emoji':  e,
                'y':       y,
                'frame':   round(t,3),
                'race':    f0['race']
            })
    return pd.DataFrame(recs)

# ─── 4) MAIN: WRITE ALL VIZS + INDEX into docs/ ─────────────────────────────

if __name__ == "__main__":
    # regression‑derived days‐saved per +1‑point improvement
    BETAS = {
        'overwhelm':   0.0379915,
        'worry':       0.1098821,
        'loneliness':  0.453555
    }

    # load & clean
    df_raw = pd.read_csv("data.csv", dtype=str)
    keep   = [
        'record_id','race_ethn_cmc',
        'lonely_freq_s2','fs_feeling_s2','fs_worry_s2','missed_days_s2'
    ]
    df      = load_and_clean(df_raw[keep])
    families = make_families(df)

    # interventions (all positive)
    scenarios = {
        'Ads Simple':     {'loneliness': 0.00},
        'Cbt':            {'loneliness': 0.27},
        'Animal Therapy': {'loneliness': 0.80},
    }

    out_files = []
    all_anim  = []

    # ensure docs/ exists
    os.makedirs("docs", exist_ok=True)

    # ─── per‑scenario animated scatter ──────────────────────────────────────
    for pretty, policy in scenarios.items():
        anim_df = build_anim_df(families, BETAS, policy, steps=30)
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
        s = fig.layout.sliders[0]
        s.currentvalue.prefix = "Progress: "
        s.currentvalue.font.size = 14

        fn = f"docs/viz_{pretty.lower().replace(' ','_')}.html"
        fig.write_html(fn, include_plotlyjs='cdn')
        print(f"→ wrote {fn}")
        out_files.append((pretty, fn))

    # ─── build time‐series of avg absences ───────────────────────────────────
    time_df = pd.concat(all_anim, ignore_index=True)
    ts = (
        time_df
        .groupby(['scenario','frame'])['y']
        .mean()
        .reset_index(name='avg_absences')
    )

    # 5a) static over‐time comparison
    fig_static = px.line(
        ts,
        x="frame", y="avg_absences",
        color="scenario",
        title="🏠 Average Absences Over Progress (by Scenario)",
        labels={"frame":"Progress","avg_absences":"Avg Absence Days"}
    )
    fig_static.update_layout(
        hovermode="x unified",
        legend_title="Scenario",
        font=dict(size=18)
    )
    stat_fn = "docs/viz_compare_time.html"
    fig_static.write_html(stat_fn, include_plotlyjs='cdn')
    print(f"→ wrote {stat_fn}")
    out_files.append(("Over-Time Comparison", stat_fn))

    # 5b) fully animated line‐drawing comparison with fixed axes
    frames = sorted(ts['frame'].unique())
    recs   = []
    for f in frames:
        subset = ts[ ts['frame'] <= f ]
        for _, row in subset.iterrows():
            recs.append({
                'scenario':    row['scenario'],
                'frame_drawn': f,
                'x':           row['frame'],
                'y':           row['avg_absences']
            })
    ts_anim = pd.DataFrame(recs)

    ymin = ts['avg_absences'].min()
    ymax = ts['avg_absences'].max()

    fig_anim = px.line(
        ts_anim,
        x="x", y="y",
        color="scenario",
        animation_frame="frame_drawn",
        title="🏠 Animated Avg Absences Over Progress",
        labels={"x":"Progress","y":"Avg Absence Days","frame_drawn":"Progress"},
        range_x=[0,1],
        range_y=[ymin, ymax]
    )
    fig_anim.update_layout(
        hovermode="x unified",
        legend_title="Scenario",
        font=dict(size=18)
    )
    fig_anim.update_xaxes(range=[0,1])
    fig_anim.update_yaxes(range=[ymin, ymax])
    s2 = fig_anim.layout.sliders[0]
    s2.currentvalue.prefix = "Progress: "
    s2.currentvalue.font.size = 14

    anim_fn = "docs/viz_compare_time_anim.html"
    fig_anim.write_html(anim_fn, include_plotlyjs='cdn')
    print(f"→ wrote {anim_fn}")
    out_files.append(("Animated Over-Time Comparison", anim_fn))

    # ─── write index.html in docs/ for GitHub Pages ─────────────────────────
    idx = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Family Absence Scenarios</title></head>",
        "<body style='font-family:sans-serif; margin:2rem;'>",
        "<h1>📊 Family Absence Scenarios</h1>",
        "<ul>"
    ]
    for label, fn in out_files:
        rel = os.path.basename(fn)
        idx.append(f"  <li><a href='{rel}' target='_blank'>{label}</a></li>")
    idx += ["</ul></body></html>"]

    with open("docs/index.html","w") as f:
        f.write("\n".join(idx))

    print("✅ All files written into docs/ (ready for GitHub Pages).")
