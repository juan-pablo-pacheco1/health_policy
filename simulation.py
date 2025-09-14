#!/usr/bin/env python3
# viz_multi_scenarios.py
#
# Summary:
# - Loads survey data, builds per-family records
# - Simulates "after-policy" absences using days-saved model + Poisson noise
# - Writes per-scenario visuals and a cross-scenario comparison
# - Writes an INTERACTIVE simulator (docs/simulator.html) – pure client-side JS (GitHub Pages friendly)
#
# Notes:
# - The "Progress" axis is a unitless 0..1 animation slider (NOT days/months/years).
# - The per-scenario scatter plots label X as "Families (anonymized)" and Y as "Absence Days".
# - The interactive page lets users tweak betas, policy improvements, race mix, Likert shifts, sample size, seed,
#   and (optionally) a mean age shift if 'child_age' exists in data.csv.

import os
import json
import copy
import numpy as np
import pandas as pd
import plotly.express as px

# ────────────────────────────────────────────────────────────────────────────
# 1. DATA CLEAN & PREP
# ────────────────────────────────────────────────────────────────────────────

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

LIKERT_COLS = ['lonely_freq_s2', 'fs_feeling_s2', 'fs_worry_s2']
CHILD_COLS = ['missed_days_s2']  # baseline absences
OPTIONAL_COLS = ['child_age']    # optional; if present, simulator exposes age shift control


def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns, drop missing, and enforce Likert 1..5 on well-being items."""
    df = df_raw.copy()
    for c in LIKERT_COLS + CHILD_COLS + [c for c in OPTIONAL_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # drop missing rows
    df = df.dropna(subset=LIKERT_COLS + CHILD_COLS)
    # we keep only values where all corresponding values for loneliness, worry, and overwhwelm are valid Likert numbers-- no negatives or > 5
    df = df[df[LIKERT_COLS].isin([1, 2, 3, 4, 5]).all(axis=1)]
    return df


def make_families(df: pd.DataFrame) -> list[dict]:
    """Return list of family dicts with numeric types and readable race strings."""
    fams = []
    for _, r in df.iterrows():
        # remove spaces
        code = str(r.get('race_ethn_cmc', '')).strip()
        # we get the race code and convert it into a readable string that expresses the actual race (not a number)
        # If code is not found, the default is "Prefer not/missing"
        race = RACE_MAP.get(code, "Prefer not/missing")
        fams.append({
            'id':         str(r['record_id']),
            'baseline':   float(r['missed_days_s2']),   # child's absence days
            'overwhelm':  float(r['fs_feeling_s2']),    # Likert 1-5
            'worry':      float(r['fs_worry_s2']),      # Likert 1-5
            'loneliness': float(r['lonely_freq_s2']),   # Likert 1-5
            'race':       race,
            # notna--> not missing
            'child_age':  float(r['child_age']) if 'child_age' in df.columns and pd.notna(r['child_age']) else None,
        })
    return fams


# ────────────────────────────────────────────────────────────────────────────
# 2. SIMULATION CORE 
# ────────────────────────────────────────────────────────────────────────────


def simulate_family(
    fam: dict,
    betas: dict,
    policy: dict,
    seed: int | None = None, # = None → means: by default, don’t use a seed unless the user provides one.
    rng: np.random.Generator | None = None
) -> dict:
    """
    Simulate a single family's 'after-policy' absence days.

    Math:
      days_saved = sum( betas[m] * policy.get(m, 0) for m )
      expected_after = max(0, baseline - days_saved)
      after ~ Poisson(lambda = expected_after)  # integer draw

      notes: 
      - "m" will take 'overhwlem', 'loneliness', and 'worry'

      - policy.get basically looks for the change in Likert points
        a policy has on 'm'. If it does not find anything, it just sets it to 0. 
        policy.get basically gets a number from "scenarios"

      - we then wanna draw from a poisson with mean = expected_after. This is 
        because otherwise every family with the same baseline would respond the 
        exact same way to the same policy. This would be unrealistically homogenious, so
        we draw from a Poisson to inject some realism. We choose a Poisson distribution
        as it allows us to sample counting data of non-negative integers, such as days 
        absent from school.

    RNG behavior:
      - Prefer passing a NumPy Generator `rng` (from np.random.default_rng(seed)).
      - If `rng` is not provided, fall back to NumPy's global RNG.
      - We *do not* call np.random.seed() here to avoid resetting global state.
    """
    # 1) Compute expected mean after-policy absences
    base = float(fam['baseline'])
    saved = sum(float(betas[m]) * float(policy.get(m, 0.0)) for m in betas)
    expct = max(0.0, base - saved)

    # 2) Draw Poisson using the provided generator if available
    if rng is not None:
        after = int(rng.poisson(lam=expct))
    else:
        after = int(np.random.poisson(lam=expct))  # falls back to global RNG

    # 3) Return a copy with the simulated 'after' value
    f2 = copy.deepcopy(fam) # make a full independent copy of the family dict
    f2['after'] = after  # add a new key 'after' to the copy
    return f2


def run_sim(
    fams: list[dict],
    betas: dict,
    policy: dict,
    seed: int | None = None
) -> list[dict]:
    """
    Simulate all families using a single Random Number Generator (RNG) seeded once for the entire run.

    - If `seed` is provided, we create a dedicated Generator with that seed.
    - Each family's Poisson draw then uses *independent* random numbers from this RNG.
    - This yields reproducible yet appropriately random outcomes per family.
    If seed=None, we don’t create a dedicated Generator.
    """

    rng = np.random.default_rng(seed) if seed is not None else None
    return [simulate_family(f, betas, policy, rng=rng) for f in fams]


# ────────────────────────────────────────────────────────────────────────────
# 3. ANIMATION DATAFRAME
# ────────────────────────────────────────────────────────────────────────────

def build_anim_df(
    fams: list[dict],       # list of family dictionaries (each contains: id, baseline days, loneliness, worry, overwhelm, race, etc.)
    betas: dict,            # regression coefficients: days saved per +1-point improvement in each domain (loneliness, worry, overwhelm)
    policy: dict,           # policy effect: expected improvement (in Likert points) for each domain, e.g., {"loneliness": 0.27}
    seed: int | None = None # optional random seed for reproducibility (controls emoji assignment and Poisson randomness)
) -> pd.DataFrame:
    """
    Build an animation-ready DataFrame using ONLY true simulation outputs:
    - frame = 0 → baseline absence days (no intervention)
    - frame = 1 → after-policy absence days (simulated with Poisson noise)

    Each family contributes exactly 2 rows:
    one for the baseline (before intervention),
    and one for the after-policy outcome.
    No tween (interpolated) frames are included.
    """

    # Run the simulation with no intervention (baseline values)
    base_run = run_sim(fams, betas, {m: 0 for m in betas}, seed=seed)

    # Run the simulation with the actual policy intervention
    pol_run  = run_sim(fams, betas, policy, seed=seed)

    # This list will collect all rows before turning into a DataFrame
    recs = []

    # Random number generator for reproducible emoji assignment
    rng = np.random.default_rng(seed)

    # Emoji pool to visually represent each family in the scatter plots
    emojis = ['👦', '👧', '🧑', '🧓', '👩']

    # Loop over families: baseline run (f0) and after-policy run (f1)
    for f0, f1 in zip(base_run, pol_run):

        # Pick one emoji to represent this family across both frames
        e = rng.choice(emojis) if seed is not None else np.random.choice(emojis)

        # --- Baseline row (frame = 0) ---
        recs.append({
            'family': f0['id'],              # unique family identifier
            'emoji':  e,                     # chosen emoji for this family
            'y':      float(f0['baseline']), # baseline absence days
            'frame':  0.0,                   # progress = 0 (start)
            'race':   f0['race'],            # race/ethnicity label
        })

        # --- After-policy row (frame = 1) ---
        recs.append({
            'family': f1['id'],              # same family identifier
            'emoji':  e,                     # same emoji as baseline
            'y':      float(f1['after']),    # simulated after-policy absence days
            'frame':  1.0,                   # progress = 1 (end)
            'race':   f1['race'],            # race/ethnicity label
        })

    # Convert list of dicts into a pandas DataFrame
    # Columns: family | emoji | y | frame | race
    return pd.DataFrame(recs)

# ────────────────────────────────────────────────────────────────────────────
# 4) INTERACTIVE SIMULATOR (pure client-side HTML+JS) — intervention only
# ────────────────────────────────────────────────────────────────────────────
def write_interactive_simulator(families: list[dict], betas: dict, out_path: str) -> None:
    """
    Interactive simulator:
      Controls: Scenario, Sample size, Seed, Trials
      Model:
        saved_days = Σ_m betas[m] * policy[m]
        λ_after    = max(0, baseline - saved_days)
        after      ~ Poisson(λ_after)
      Runs multiple trials → shows averages with error bars and full distribution.
    """
    families_min = [{
        "baseline":   float(f["baseline"]),
        "loneliness": float(f["loneliness"]),
        "worry":      float(f["worry"]),
        "overwhelm":  float(f["overwhelm"]),
    } for f in families]

    scenarios = {
        "No Intervention": {"loneliness": 0.0,  "worry": 0.0, "overwhelm": 0.0},
        "Ads Simple":      {"loneliness": 0.00, "worry": 0.0, "overwhelm": 0.0},
        "Cbt":             {"loneliness": 0.27, "worry": 0.0, "overwhelm": 0.0},
        "Animal Therapy":  {"loneliness": 0.80, "worry": 0.0, "overwhelm": 0.0},
    }

    html = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Family Absence Simulator</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.25rem;}
    .row{display:flex;gap:1rem;flex-wrap:wrap;}
    .card{border:1px solid #ddd;border-radius:10px;padding:1rem;flex:1;min-width:300px;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:.5rem 1rem;align-items:center;}
    select,input{padding:.35rem;border:1px solid #ccc;border-radius:6px;}
    button{padding:.5rem 1rem;border:1px solid #ccc;border-radius:8px;background:#fafafa;cursor:pointer;}
    button:hover{background:#f0f0f0;}
    h1{margin:.2rem 0 1rem;}
  </style>
</head>
<body>
  <h1>🏠 Family Absence Simulator</h1>
  <p style="max-width:720px;color:#444;">
    Choose an intervention and simulate its effect on absence days. Results show averages with error bars (uncertainty)
    and the full distribution of simulated absence days. This helps policymakers see both expected impacts and variability.
  </p>

  <div class="row">
    <div class="card" style="max-width:420px;">
      <h2>Settings</h2>
      <div class="grid">
        <div>Scenario</div>
        <select id="scenario"></select>

        <div>Sample size</div>
        <input id="sample_n" type="number" min="10" step="10">

        <div>Seed</div>
        <input id="seed" type="number" step="1" value="1234">

        <div>Trials</div>
        <input id="trials" type="number" min="1" step="1" value="30">
      </div>
      <div style="margin-top:.75rem;">
        <button id="btn_run">Run Simulation</button>
      </div>
    </div>

    <div class="card">
      <h2>Average Absences (± error)</h2>
      <div id="avg_chart" style="height:300px;"></div>
    </div>
  </div>

  <div class="row" style="margin-top:1rem;">
    <div class="card">
      <h2>Distribution: Baseline vs After</h2>
      <div id="hist_chart" style="height:320px;"></div>
    </div>
  </div>

<script>
  const FAMILIES  = %%FAMILIES_JSON%%;
  const BETAS     = %%BETAS_JSON%%;
  const SCENARIOS = %%SCENARIOS_JSON%%;

  function mulberry32(a){return function(){var t=a+=0x6D2B79F5;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;}}
  function poisson(lam,rng){ if(lam<=0) return 0; const L=Math.exp(-lam); let k=0,p=1; do{ k++; p*=rng(); }while(p>L); return k-1; }

  function sampleFamilies(N, rng){
    const out=[]; for(let i=0;i<N;i++){ const idx=Math.floor(rng()*FAMILIES.length); out.push(FAMILIES[idx]); }
    return out;
  }

  function getSettings(){
    const scenName=document.getElementById('scenario').value;
    const N=Math.max(10, parseInt(document.getElementById('sample_n').value||"10",10));
    const seed=parseInt(document.getElementById('seed').value||"1234",10);
    const trials=Math.max(1, parseInt(document.getElementById('trials').value||"30",10));
    return { scenName, policy:SCENARIOS[scenName], N, seed, trials };
  }

  function simulate(policy, N, seed, trials){
    const baselineAvgs=[], afterAvgs=[], baselineAll=[], afterAll=[];
    for(let t=0;t<trials;t++){
      const rng=mulberry32(seed+t);
      const sample=sampleFamilies(N,rng);
      const saved=(BETAS.loneliness||0)*(policy.loneliness||0)+(BETAS.worry||0)*(policy.worry||0)+(BETAS.overwhelm||0)*(policy.overwhelm||0);
      const baseline=[], after=[];
      for(const f of sample){
        const base=Number(f.baseline);
        const lam=Math.max(0, base - saved);
        baseline.push(base); after.push(poisson(lam,rng));
      }
      baselineAvgs.push(baseline.reduce((a,b)=>a+b,0)/baseline.length);
      afterAvgs.push(after.reduce((a,b)=>a+b,0)/after.length);
      baselineAll.push(...baseline); afterAll.push(...after);
    }
    function mean(a){return a.reduce((x,y)=>x+y,0)/a.length;}
    function std(a){const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);}
    return {
      baselineAvg:mean(baselineAvgs), afterAvg:mean(afterAvgs),
      baselineErr:std(baselineAvgs), afterErr:std(afterAvgs),
      baselineArr:baselineAll, afterArr:afterAll
    };
  }

  function render(r, scenName, trials){
    Plotly.react('avg_chart', [{
      x:['Baseline','After ('+scenName+')'],
      y:[r.baselineAvg, r.afterAvg],
      error_y:{type:'data', array:[r.baselineErr, r.afterErr], visible:true},
      type:'bar',
      text:[r.baselineAvg.toFixed(2), r.afterAvg.toFixed(2)],
      textposition:'auto'
    }], { yaxis:{title:'Absence Days'}, title:'Based on '+trials+' trials' }, {displayModeBar:false});

    Plotly.react('hist_chart', [
      { x:r.baselineArr, type:'histogram', opacity:.6, name:'Baseline' },
      { x:r.afterArr,    type:'histogram', opacity:.6, name:'After ('+scenName+')' }
    ], { barmode:'overlay', xaxis:{title:'Absence Days'}, yaxis:{title:'Families'}, legend:{orientation:'h'} }, {displayModeBar:false});
  }

  function run(){ const s=getSettings(); const r=simulate(s.policy,s.N,s.seed,s.trials); render(r,s.scenName,s.trials); }
  function init(){
    const sel=document.getElementById('scenario');
    for(const name of Object.keys(SCENARIOS)){ const o=document.createElement('option'); o.value=name; o.textContent=name; sel.appendChild(o); }
    document.getElementById('sample_n').value=FAMILIES.length;
    document.getElementById('btn_run').addEventListener('click',run);
    run();
  }
  document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""

    import json, os
    html = html.replace("%%FAMILIES_JSON%%", json.dumps(families_min))
    html = html.replace("%%BETAS_JSON%%", json.dumps(betas))
    html = html.replace("%%SCENARIOS_JSON%%", json.dumps(scenarios))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# ────────────────────────────────────────────────────────────────────────────
# 5. MAIN (KEEPING YOUR REQUIRED BLOCK VERBATIM)
# ────────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────────
# 6) POST-BLOCK AUGMENTATIONS (do NOT modify the block above)
# ────────────────────────────────────────────────────────────────────────────

    # (A) Add "No Intervention" scenario without editing your verbatim block
    scenarios = {'No Intervention': {}} | scenarios  # requires Python 3.9+

    # (B) Prepare outputs
    out_files: list[tuple[str, str]] = []
    all_anim: list[pd.DataFrame] = []
    os.makedirs("docs", exist_ok=True)

    # (C) Per-scenario animated scatter with clear axis labels
    for pretty, policy in scenarios.items():
        anim_df = build_anim_df(families, BETAS, policy, seed=42)
        anim_df['scenario'] = pretty
        all_anim.append(anim_df)

        title = f"🏠 Family Absence Game: Before → After Policy ({pretty})"
        fig = px.scatter(
            anim_df,
            x="family", y="y",
            animation_frame="frame",
            text="emoji",
            range_y=[-0.5, float(anim_df['y'].max()) + 1],
            title=title,
            hover_data=['family','emoji','y','frame','race']
        )
        fig.update_traces(textposition="middle center", marker=dict(size=1, opacity=0))
        fig.update_layout(
            xaxis=dict(showticklabels=False, title="Families (anonymized)"),
            yaxis=dict(title="Absence Days"),
            font=dict(size=18)
        )
        if getattr(fig.layout, "sliders", None):
            s = fig.layout.sliders[0]
            s.currentvalue.prefix = "Progress (0→1, unitless): "
            s.currentvalue.font.size = 14

        fn = f"docs/viz_{pretty.lower().replace(' ','_')}.html"
        fig.write_html(fn, include_plotlyjs='cdn')
        print(f"→ wrote {fn}")
        out_files.append((pretty, fn))

    # (D) Time-series of average absences
    time_df = pd.concat(all_anim, ignore_index=True)
    ts = (
        time_df
        .groupby(['scenario','frame'])['y']
        .mean()
        .reset_index(name='avg_absences')
    )

    # Static comparison (clear labels)
    fig_static = px.line(
        ts,
        x="frame", y="avg_absences",
        color="scenario",
        title="🏠 Average Absences Over Progress (by Scenario)",
        labels={"frame":"Progress (0→1, unitless)","avg_absences":"Avg Absence Days"}
    )
    fig_static.update_layout(
        hovermode="x unified",
        legend_title="Scenario",
        font=dict(size=18),
        xaxis_title="Progress (0→1, unitless)",
        yaxis_title="Avg Absence Days"
    )
    stat_fn = "docs/viz_compare_time.html"
    fig_static.write_html(stat_fn, include_plotlyjs='cdn')
    print(f"→ wrote {stat_fn}")
    out_files.append(("Over-Time Comparison", stat_fn))

    # Animated line drawing (labels clarify unitless "Progress")
    frames = sorted(ts['frame'].unique())
    recs = []
    for f in frames:
        subset = ts[ts['frame'] <= f]
        for _, row in subset.iterrows():
            recs.append({
                'scenario':    row['scenario'],
                'frame_drawn': f,
                'x':           row['frame'],
                'y':           row['avg_absences']
            })
    ts_anim = pd.DataFrame(recs)

    ymin = float(ts['avg_absences'].min())
    ymax = float(ts['avg_absences'].max())

    fig_anim = px.line(
        ts_anim,
        x="x", y="y",
        color="scenario",
        animation_frame="frame_drawn",
        title="🏠 Animated Avg Absences Over Progress",
        labels={"x":"Progress (0→1, unitless)","y":"Avg Absence Days","frame_drawn":"Progress (0→1)"},
        range_x=[0,1],
        range_y=[ymin, ymax]
    )
    fig_anim.update_layout(
        hovermode="x unified",
        legend_title="Scenario",
        font=dict(size=18),
        xaxis_title="Progress (0→1, unitless)",
        yaxis_title="Avg Absence Days"
    )
    fig_anim.update_xaxes(range=[0,1])
    fig_anim.update_yaxes(range=[ymin, ymax])
    if getattr(fig_anim.layout, "sliders", None):
        s2 = fig_anim.layout.sliders[0]
        s2.currentvalue.prefix = "Progress (0→1, unitless): "
        s2.currentvalue.font.size = 14

    anim_fn = "docs/viz_compare_time_anim.html"
    fig_anim.write_html(anim_fn, include_plotlyjs='cdn')
    print(f"→ wrote {anim_fn}")
    out_files.append(("Animated Over-Time Comparison", anim_fn))

    # (E) Interactive simulator page
    sim_fn = "docs/simulator.html"
    write_interactive_simulator(families, BETAS, sim_fn)
    print(f"→ wrote {sim_fn}")
    out_files.append(("Interactive Simulator", sim_fn))

    # (F) Minimal index
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
    idx += [
        "</ul>",
        "<p style='color:#555;max-width:720px'>",
        "Notes: (1) The X-axis labeled “Progress (0→1)” is a unitless animation slider (not days/months/years). ",
        "(2) Per-scenario scatters use X = Families (anonymized), Y = Absence Days. ",
        "(3) Try the Interactive Simulator to tweak demographics and policy settings.",
        "</p>",
        "</body></html>"
    ]
    with open("docs/index.html","w", encoding="utf-8") as f:
        f.write("\n".join(idx))

    print("✅ All files written into docs/ (ready for GitHub Pages).")
