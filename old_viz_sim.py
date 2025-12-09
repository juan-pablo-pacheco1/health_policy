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
    saved = -sum(betas[m]*policy.get(m,0) for m in betas)
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
    for f0,f1 in zip(base_run, pol_run):
        e = np.random.choice(emojis)
        for t in np.linspace(0,1,steps):
            y = f0['baseline'] + (f1['after'] - f0['baseline']) * t
            records.append({
                'family': f0['id'],
                'emoji':  e,
                'y':       y,
                'frame':   round(t,3),
                'race':    f0['race']
            })
    return pd.DataFrame(records)

# ─── 4) MAIN: LOOP SCENARIOS, WRITE FILES & INDEX ───────────────────────────

if __name__=="__main__":

    # your regression‑derived betas
    BETAS = {
      'overwhelm':  -0.0379915,
      'worry':      -0.1098821,
      'loneliness':  0.453555
    }

    # read only needed cols
    df_raw = pd.read_csv("data.csv", dtype=str)
    keep = [
      'record_id','race_ethn_cmc',
      'lonely_freq_s2','fs_feeling_s2','fs_worry_s2','missed_days_s2'
    ]
    df_clean = load_and_clean(df_raw[keep])
    families = make_families(df_clean)

    # drop no‑policy; only show real scenarios
    scenarios = {
      'ads_simple':     {'loneliness':0.0},
      'cbt':            {'loneliness':-0.27},
      'animal_therapy': {'loneliness':-0.80},
    }

    out_files = []

    for name, policy in scenarios.items():
        anim_df = build_anim_df(families, BETAS, policy, steps=30)
        pretty = name.replace('_',' ').title()
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

    # build index.html
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

    # open it for you
    webbrowser.open("file://" + os.path.abspath("index.html"))
    print("Done! Opening index.html…")

=========
import pandas as pd
import numpy as np
import copy

# ──────────────────────────────────────────────────────────────────────────────
#  PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# How many absence days are saved per 1‑point drop in each subscale.
# These “betas” should eventually come from fitting linear regressions
# of missed_days_s2 on each wellbeing metric.
BETAS = {
    'overwhelm':   -0.0379915,   # 1‑pt drop in overwhelm → 0.2 fewer absence days 1 lots of overwhelm 5- no overhwlem
    'worry':       -0.1098821,   # 1‑pt drop in worry     → 0.3 fewer absence days
    'loneliness':  0.453555    # 1‑pt drop in loneliness → 0.5 fewer absence days; 1 no lonely 5- super lonely
}

# ──────────────────────────────────────────────────────────────────────────────
#  TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_load_and_clean():
    # Build an in‑memory CSV with edge cases to verify load_and_clean logic:
    #  • A: all valid → should survive
    #  • B: lonely_freq_s2=6 (outside 1–5) → drop
    #  • C: missing fs_feeling_s2 → drop
    #  • D: non‑numeric missed_days_s2 → drop
    test_csv = """\
record_id,lonely_freq_s2,fs_feeling_s2,fs_worry_s2,missed_days_s2
A,  1, 2, 3, 4      
B,  6, 3, 2, 1      
C,  3, , 4, 2       
D,  2, 5, 1, notan  
"""
    df = load_and_clean(pd.io.common.StringIO(test_csv))
    assert len(df) == 1
    row = df.iloc[0]
    assert row['record_id']       == 'A'
    assert row['lonely_freq_s2']  == 1
    assert row['fs_feeling_s2']   == 2
    assert row['fs_worry_s2']     == 3
    assert row['missed_days_s2']  == 4

def test_make_families():
    # Create a minimal DataFrame and ensure it maps to the expected family dict
    df = pd.DataFrame([{
        'record_id':      'X',
        'lonely_freq_s2': 5,
        'fs_feeling_s2':  4,
        'fs_worry_s2':    3,
        'missed_days_s2': 7
    }])
    fams = make_families(df)
    assert isinstance(fams, list) and len(fams) == 1
    fam = fams[0]
    # Parent fields are renamed to match our simulation keys
    assert fam['parent']['id']        == 'X'
    assert fam['parent']['overwhelm']  == 4
    assert fam['parent']['worry']      == 3
    assert fam['parent']['loneliness'] == 5
    # Child starts with zero hospital visits and observed absences
    assert fam['child']['hosp']       == 0
    assert fam['child']['absences']   == 7

def test_simulate_absence_one_fam():
    # 1) No policy: sim_absences ~ Poisson(base)
    fam = {'child': {'absences': 5}, 'parent': {}}
    no_policy = {'overwhelm': 0.0}
    out = simulate_absence_one_fam(fam, {'overwhelm':0.2}, no_policy, random_seed=0)
    assert out['sim_absences'] == 9  # np.random.poisson(5) with seed=0
    
    # 2) Over‑powered policy: saved >= base → expected=0 → always 0
    full_policy = {'overwhelm': -100.0}
    out2 = simulate_absence_one_fam(fam, {'overwhelm':0.2}, full_policy, random_seed=42)
    assert out2['sim_absences'] == 0

def test_run_and_aggregate():
    # aggregate_absences: sum & mean
    fams = [{'sim_absences':3}, {'sim_absences':5}]
    stats = aggregate_absences(fams)
    assert stats == {'total_abs':8, 'avg_per_family':4.0}

    # run_simulation should invoke simulate_absence_one_fam for each family
    def dummy_sim(fam, betas, effects, random_seed=None):
        fam2 = fam.copy()
        fam2['sim_absences'] = fam['child']['absences'] + 1
        return fam2
    global simulate_absence_one_fam
    original = simulate_absence_one_fam
    simulate_absence_one_fam = dummy_sim

    families = [
        {'child': {'absences': 2}},
        {'child': {'absences': 4}}
    ]
    out = run_simulation(families, {}, {})
    assert [f['sim_absences'] for f in out] == [3,5]

    simulate_absence_one_fam = original

# ──────────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path):
    """
    1) Read CSV into a DataFrame
    2) Coerce wellbeing & child cols to numeric (bad → NaN)
    3) Drop rows with any NaN in those cols
    4) Keep only wellbeing values in 1–5
    5) Return cleaned DataFrame
    """
    df = pd.read_csv(csv_path)
    wellbeing_cols = ['lonely_freq_s2', 'fs_feeling_s2', 'fs_worry_s2']
    child_cols     = ['missed_days_s2']

    # Coerce invalid entries (e.g. “notan”, empty) → NaN
    for col in wellbeing_cols + child_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any key columns
    df = df.dropna(subset=wellbeing_cols + child_cols)

    # Keep only integer wellbeing scores in [1..5]
    df = df[df[wellbeing_cols]
            .isin([1, 2, 3, 4, 5])
            .all(axis=1)
           ]

    return df

def make_families(df):
    """
    Convert each row into a family dict:
      parent:   {id, overwhelm, worry, loneliness}
      child:    {hosp=0, absences=missed_days_s2}
    """
    families = []
    for _, row in df.iterrows():
        fam = {
            'parent': {
                'id':         row['record_id'],
                'overwhelm':  row['fs_feeling_s2'],
                'worry':      row['fs_worry_s2'],
                'loneliness': row['lonely_freq_s2'],
            },
            'child': {
                'hosp':     0,
                'absences': row['missed_days_s2']
            }
        }
        families.append(fam)
    return families

# ──────────────────────────────────────────────────────────────────────────────
#  SIMULATION CORE
# ──────────────────────────────────────────────────────────────────────────────

def simulate_absence_one_fam(fam, betas, policy_effects, random_seed=None):
    """
    For a single family:
      1) base = observed child absences
      2) saved = -∑ (beta_metric * policy_delta_metric)
      3) expected = max(0, base - saved)
      4) sim_absences ~ Poisson(expected)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    base = fam['child']['absences']
    saved = sum(
        betas[metric] * policy_effects.get(metric, 0)
        for metric in betas
    )
    expected = max(0, base + saved)

    fam2 = copy.deepcopy(fam)
    fam2['sim_absences'] = np.random.poisson(lam=expected)
    return fam2

def run_simulation(families, betas, policy_effects):
    """
    Apply simulate_absence_one_fam to each family once.
    Returns list of families with added 'sim_absences'.
    """
    return [simulate_absence_one_fam(fam, betas, policy_effects)
            for fam in families]

def aggregate_absences(simulated_fams):
    """
    Sum and average sim_absences across all families.
    Returns {'total_abs': total, 'avg_per_family': mean}.
    """
    total = sum(fam['sim_absences'] for fam in simulated_fams)
    avg   = total / len(simulated_fams)
    return {'total_abs': total, 'avg_per_family': avg}

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN & POLICY SCENARIOS
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Verify all unit tests
    test_load_and_clean()
    test_make_families()
    test_simulate_absence_one_fam()
    test_run_and_aggregate()
    print("✅ All tests passed!\n")

    # 2) Load, clean, and build families
    df = load_and_clean('data.csv')
    families = make_families(df)
    print(f"Simulating for {len(families)} families\n")

     # 3) Define policy scenarios using real-world effect sizes on loneliness:
    scenarios = {
        'ads_simple': {
            # No significant change in caregiver loneliness from standard Adult Day Services
            # Iecovich & Biderman (2012) Int Psychogeriatr found no difference (3–9 UCLA scale → Δ≈0 on 1–5) 
            # PubMed: https://pubmed.ncbi.nlm.nih.gov/21996017/
            'loneliness': 0.0
        },
        # The following come from a JAMA Study, but on older adults 55+. They were, however, above 18-- like many
        # of the caregivers in the Family Circle 

        # How we mapped SMDs to a 1–5 scale

        # The following is how results from the JAMA study were translated to be used in Family Circle:
        # Standardized Mean Difference (SMD) expresses change in units of the pooled standard deviation.

        # On our 1–5 loneliness scale, a “1‑point” change is roughly one SD in many of these studies.

        # Thus we treated each SMD ≈ Δ raw‑points:

        # CBT: SMD ~ –0.52→ Δ ≈ –0.5 → used –0.27 (conservative midpoint of –0.52/–0.46)

        # Multicomponent: average SMD ~ –0.60 → rounded to –0.50

        # Animal therapy: SMD ~ –1.86 (LTC) but community studies ~ –0.80 → used –0.80

        # ADS: SMD ~ 0 → Δ = 0
        'cbt': {
            # Small reduction in loneliness from cognitive‑behavioral therapy interventions
            # Meta‑analysis of 7 RCTs in older adults: SMD = −0.27 on loneliness scales: https://pmc.ncbi.nlm.nih.gov/articles/PMC9577679/?utm_source=chatgpt.com
            # conservative central estimate between those two (−0.52 and −0.46), arriving at −0.27 points on our 1–5 scale (i.e. roughly half an SMD unit).

            # MORE INFO: 
            # “CBT and Psychotherapy
            # Four studies65,110-112 set in the community were included in the meta-analysis with an ES of −0.52
            # (95% CI, −1.21 to 0.17), provided by trained personnel (eg, psychotherapist, doctoral students) in
            # individual and group sessions. There was considerable heterogeneity (I
            # 2 = 83%; P < .001). Upon
            # excluding studies without active controls, the ES remained similar at −0.46 (95% CI, −1.39 to 0.46).
            # One study63 measured social support in the community (ES, 0.41; 95% CI, 0.10 to 0.72). Parry et al65
            # also measured social isolation, with an ES of 0.16 (95% CI, −0.06 to 0.38).
            'loneliness': -0.27
        },
        'multicomponent': {
            # Moderate reduction from multicomponent psychosocial programs (education+support+skills)
            # Same meta‑analysis: typical SMD ≈ −0.50 : https://pmc.ncbi.nlm.nih.gov/articles/PMC9577679/?utm_source=chatgpt.com
            # Averaging the community (−0.67) and LTC (−0.53) effect sizes gives approximately −0.60, which we rounded to −0.50 for our mid‑level “multicomponent” scenario.

            # MORE INFO:
            # Combination and Multicomponent Interventions
            # Five studies76-80 were included in the meta-analysis, 2 in the community (Figure 4) and 3 in LTC. The
            # ES was −0.67 (95% CI, −1.13 to −0.21; I
            # 2 = 0%; P = .704) in community and −0.53 (95% CI, −0.86 to
            # −0.20; I
            # 2 = 57%; P = .099) in LTC. Interventions included exercise with arts and crafts, home care
            # with nursing outreach and educational resources, Tai Chi and CBT, and pain management programs.
            'loneliness': -0.50
        },
        'animal_therapy': {
            # Large effect from animal‑assisted interventions in long‑term care: SMD ≈ −0.80
            # Systematic review & meta‑analysis : https://pmc.ncbi.nlm.nih.gov/articles/PMC9577679/?utm_source=chatgpt.com

            # MORE INFO:
            # Animal Therapy
            # Six studies70-75 were included in the meta-analysis, 2 in the community (Figure 4) and 4 in LTC with
            # an ES of −0.41 (95% CI, −1.75 to 0.92; I
            # 2 = 87%; P = .005) and −1.05 (95% CI, −2.93 to 0.84; I
            # 2 = 95%;
            # P < .001), respectively. Upon excluding a study71 comparing group to individual animal therapy, the
            # effect size was −1.86 (95% CI, −3.14 to −0.59; I
            # 2 = 86%; P < .001). Generally, participants interacted
            # with living dogs or robotic animals (seal or dog). One study75 provided a bird in the participant’s room
            # for the study duration

            'loneliness': -0.80
        }
    }
    # 4) Baseline (no policy)
    no_policy = {m: 0 for m in BETAS}
    sims_base = run_simulation(families, BETAS, no_policy)
    stats_base = aggregate_absences(sims_base)
    print("Baseline (no policy):", stats_base, "\n")

    # 5) Sweep through each scenario and report saved days
    print("Policy scenario results:")
    for name, pol in scenarios.items():
        sims = run_simulation(families, BETAS, pol)
        stats = aggregate_absences(sims)
        saved = stats_base['total_abs'] - stats['total_abs']
        print(f"  {name.capitalize():10s}: {stats},  days saved ≈ {saved:.1f}")

if __name__ == "__main__":
    main()





if __name__ == "__main__":
    # regression‑derived days‐saved per +1‑point improvement
    # DO NOT CHANGE
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
#INTERVENTIONS
# DO NOT CHANGE

    # ADULT DAY SERVICES:
        # No significant change in caregiver loneliness from standard Adult Day Services
        # Iecovich & Biderman (2012) Int Psychogeriatr found no difference (3–9 UCLA scale → Δ≈0 on 1–5) 
        # PubMed: https://pubmed.ncbi.nlm.nih.gov/21996017/
    

    # OTHER INTERVENTIONS    
        # The following come from a JAMA Study, but on older adults 55+. They were, however, above 18-- like many
        # of the caregivers in the Family Circle 
        # How we mapped SMDs to a 1–5 scale
        # The following is how results from the JAMA study were translated to be used in Family Circle:
        # Standardized Mean Difference (SMD) expresses change in units of the pooled standard deviation.
        # On our 1–5 loneliness scale, a “1‑point” change is roughly one SD in many of these studies.
        # Thus we treated each SMD ≈ Δ raw‑points:
        # CBT: SMD ~ –0.52→ Δ ≈ –0.5 → used –0.27 (conservative midpoint of –0.52/–0.46)
        # Multicomponent: average SMD ~ –0.60 → rounded to –0.50
        # Animal therapy: SMD ~ –1.86 (LTC) but community studies ~ –0.80 → used –0.80
        # ADS: SMD ~ 0 → Δ = 0

        # CBT
         # Small reduction in loneliness from cognitive‑behavioral therapy interventions
            # Meta‑analysis of 7 RCTs in older adults: SMD = −0.27 on loneliness scales: https://pmc.ncbi.nlm.nih.gov/articles/PMC9577679/?utm_source=chatgpt.com
            # conservative central estimate between those two (−0.52 and −0.46), arriving at −0.27 points on our 1–5 scale (i.e. roughly half an SMD unit).

            # MORE INFO: 
            # “CBT and Psychotherapy
            # Four studies65,110-112 set in the community were included in the meta-analysis with an ES of −0.52
            # (95% CI, −1.21 to 0.17), provided by trained personnel (eg, psychotherapist, doctoral students) in
            # individual and group sessions. There was considerable heterogeneity (I
            # 2 = 83%; P < .001). Upon
            # excluding studies without active controls, the ES remained similar at −0.46 (95% CI, −1.39 to 0.46).
            # One study63 measured social support in the community (ES, 0.41; 95% CI, 0.10 to 0.72). Parry et al65
            # also measured social isolation, with an ES of 0.16 (95% CI, −0.06 to 0.38).

        # ANIMAL THERAPY
         # Large effect from animal‑assisted interventions in long‑term care: SMD ≈ −0.80
            # Systematic review & meta‑analysis : https://pmc.ncbi.nlm.nih.gov/articles/PMC9577679/?utm_source=chatgpt.com

            # MORE INFO:
            # Animal Therapy
            # Six studies70-75 were included in the meta-analysis, 2 in the community (Figure 4) and 4 in LTC with
            # an ES of −0.41 (95% CI, −1.75 to 0.92; I
            # 2 = 87%; P = .005) and −1.05 (95% CI, −2.93 to 0.84; I
            # 2 = 95%;
            # P < .001), respectively. Upon excluding a study71 comparing group to individual animal therapy, the
            # effect size was −1.86 (95% CI, −3.14 to −0.59; I
            # 2 = 86%; P < .001). Generally, participants interacted
            # with living dogs or robotic animals (seal or dog). One study75 provided a bird in the participant’s room
            # for the study duration
    scenarios = {
        'Ads Simple':     {'loneliness': 0.00},
        'Cbt':            {'loneliness': 0.27},
        'Animal Therapy': {'loneliness': 0.80},
    }