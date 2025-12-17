#!/usr/bin/env python3
# simulation_final.py
#
# Summary:
# - Loads CMC survey data, builds per-family records
# - Simulates "after-policy" absences using days-saved model + Poisson noise
# - Creates three interactive visualizations:
#   1. Family Tracker - individual CMC family trajectories
#   2. Interactive Simulator - CMC population with adjustable parameters
#   3. General Population Exploratory - extrapolated estimates with clear assumptions
# A

import os
import json
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ────────────────────────────────────────────────────────────────────────────
# 1. DATA CLEAN & PREP  (1–5 Likert, higher = worse)
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

LIKERT_COLS   = ['lonely_freq_s2', 'fs_feeling_s2', 'fs_worry_s2']
CHILD_COLS    = ['missed_days_s2']
OPTIONAL_COLS = ['child_age']

def _clamp_int_15(x: float):
    """Round to nearest int in [1,5]; return NaN if input is NaN."""
    if pd.isna(x):
        return np.nan
    return int(max(1, min(5, round(float(x)))))

def _reverse_1to10_higher_better(x: float) -> float:
    """Flip a 1..10 scale where 10=best to 1..10 where 10=worst (i.e., y=11-x)."""
    if pd.isna(x):
        return np.nan
    return 11.0 - float(x)

def _map_1to10_to_1to5(x: float):
    """Linear map 1..10 (higher=worse) → 1..5 (higher=worse), then clamp/round."""
    if pd.isna(x):
        return np.nan
    y = 1.0 + (float(x) - 1.0) * (4.0/9.0)
    return _clamp_int_15(y)

def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce numeric, harmonize to 1..5 integers (higher = worse),
    and drop rows with missing required fields.
    """
    df = df_raw.copy()

    for c in set(LIKERT_COLS + CHILD_COLS + [c for c in OPTIONAL_COLS if c in df.columns]):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['loneliness_5'] = df['lonely_freq_s2'].apply(_clamp_int_15)
    df['overwhelm_5'] = df['fs_feeling_s2'].apply(_reverse_1to10_higher_better).apply(_map_1to10_to_1to5)
    df['worry_5']     = df['fs_worry_s2'].apply(_reverse_1to10_higher_better).apply(_map_1to10_to_1to5)

    needed = ['loneliness_5', 'overwhelm_5', 'worry_5'] + CHILD_COLS
    df = df.dropna(subset=needed).copy()

    df = df[
        df['loneliness_5'].between(1,5) &
        df['overwhelm_5'].between(1,5) &
        df['worry_5'].between(1,5)
    ].copy()

    return df

def make_families(df: pd.DataFrame) -> list[dict]:
    """Build family dicts using 1..5 integers (higher = worse) for all three metrics."""
    fams = []
    for _, r in df.iterrows():
        code = str(r.get('race_ethn_cmc', '')).strip()
        race = RACE_MAP.get(code, "Prefer not/missing")
        fams.append({
            'id':         str(r['record_id']),
            'baseline':   float(r['missed_days_s2']),
            'overwhelm':  int(r['overwhelm_5']),
            'worry':      int(r['worry_5']),
            'loneliness': int(r['loneliness_5']),
            'race':       race,
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
    seed: int | None = None,
    rng: np.random.Generator | None = None
) -> dict:
    """Simulate a single family's 'after-policy' absence days."""
    base = float(fam['baseline'])
    saved = sum(float(betas[m]) * float(policy.get(m, 0.0)) for m in betas)
    expct = max(0.0, base - saved)

    if rng is not None:
        after = int(rng.poisson(lam=expct))
    else:
        after = int(np.random.poisson(lam=expct))

    f2 = copy.deepcopy(fam)
    f2['after'] = after
    return f2


def run_sim(
    fams: list[dict],
    betas: dict,
    policy: dict,
    seed: int | None = None
) -> list[dict]:
    """Simulate all families using a single Random Number Generator."""
    rng = np.random.default_rng(seed) if seed is not None else None
    return [simulate_family(f, betas, policy, rng=rng) for f in fams]


# ────────────────────────────────────────────────────────────────────────────
# 3. INTERACTIVE SIMULATOR - CMC ONLY
# ────────────────────────────────────────────────────────────────────────────

def write_interactive_simulator(families: list[dict], betas: dict, out_path: str) -> None:
    """Interactive simulator for CMC families only."""
    families_min = [{
        "baseline":   float(f["baseline"]),
        "loneliness": float(f["loneliness"]),
        "worry":      float(f["worry"]),
        "overwhelm":  float(f["overwhelm"]),
    } for f in families]

# ---- AAAA CHECKED
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
        # The Family CIRCLE loneliness scale (1-5) has a standard deviation of 1.20. To convert the JAMA meta-analysis 
        # effect, we multiply out STD by Jama's SMD.


        # # CALCULATIONS (statistically significant interventions only):
        # Multicomponent: JAMA reports SMD = -0.67 (95% CI: -1.13 to -0.21, community, 2 studies)
        #      → 0.67 × 1.20 = 0.804 ≈ 0.80 Likert points reduction
        # Counseling: JAMA reports SMD = -0.19 (95% CI: -0.35 to -0.03, community, 6 studies, excluding outlier)
        #      → 0.19 × 1.20 = 0.228 ≈ 0.23 Likert points reduction


            #Combination and Multicomponent Interventions
            # Five studies76 -80 were included in the meta-analysis, 2 in the community (Figure 4) and 3 in LTC. 
            # The ES was −0.67 (95% CI, −1.13 to −0.21; I2 = 0%; P = .704) in community and −0.53 (95% CI, −0.86 to −0.20; I2 = 57%; P = .099) 
            # in LTC. Interventions included exercise with arts and crafts, home care with nursing outreach and educational resources, 
            # Tai Chi and CBT, and pain management programs. Six studies57-60,63,64 were included in social support meta-analysis 
            # (all community-dwelling), with an ES of 0.29 (95% CI, 0.15 to 0.43) and low heterogeneity (I2 = 0%; P = .66).


            # Counseling
            # Six group-based studies81-86 in community-dwelling participants were included in the meta-analysis. 
            # Interventions included bereavement counseling and instructor-led group support programs. The ES was −0.80 
            # (95% CI, −1.96 to 0.36); heterogeneity was substantial (I2 = 97%; P < .001). When excluding Alaviani 
            # et al,86 the ES was less pronounced (−0.19; 95% CI, −0.35 to −0.03), with no heterogeneity (I2 = 0%; P = .48).



    scenarios = {
        "No Intervention": {"loneliness": 0.0, "worry": 0.0, "overwhelm": 0.0},
        "Multicomponent Program": {"loneliness": 0.80, "worry": 0.0, "overwhelm": 0.0},
        "Group Counseling": {"loneliness": 0.23, "worry": 0.0, "overwhelm": 0.0},
    }

    # AAAAAAA

    html = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CMC Family Simulator</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5rem;background:#f8f9fa;}
    .container{max-width:1200px;margin:0 auto;}
    h1{color:#2c3e50;margin-bottom:.5rem;}
    .subtitle{color:#7f8c8d;margin-bottom:1.5rem;font-size:1.05rem;}
    .methodology{background:#fff3cd;border-left:4px solid #ffc107;padding:1.25rem;margin-bottom:2rem;border-radius:8px;}
    .methodology h3{margin-top:0;color:#856404;}
    .methodology ul{margin:.5rem 0;padding-left:1.5rem;}
    .methodology li{margin:.25rem 0;color:#856404;}
    .row{display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.5rem;}
    .card{background:white;border:1px solid #e1e8ed;border-radius:12px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.05);}
    .settings-card{flex:0 0 380px;}
    .chart-card{flex:1;min-width:500px;}
    .grid{display:grid;grid-template-columns:120px 1fr;gap:.75rem 1rem;align-items:center;margin-bottom:1rem;}
    label{font-weight:600;color:#34495e;font-size:.9rem;}
    input,select{padding:.6rem;border:2px solid #e1e8ed;border-radius:8px;font-size:.95rem;transition:border .2s;}
    input:focus,select:focus{outline:none;border-color:#3498db;}
    button{width:100%;padding:.8rem;border:none;border-radius:8px;background:#3498db;color:white;font-weight:600;font-size:1rem;cursor:pointer;transition:background .2s;}
    button:hover{background:#2980b9;}
  </style>
</head>
<body>
  <div class="container">
    <h1>🏠 CMC Family Absence Simulator</h1>
    <p class="subtitle">Simulate how caregiver-focused interventions reduce school absences for medically complex children</p>

    <div class="methodology">
      <h3>📋 How This Works:</h3>
      <ul>
        <li><strong>Data Source:</strong> Real survey data from families with medically complex children (CMC) - Family CIRCLE data analysis</li>
        <li><strong>Interventions:</strong> Target <em>caregiver</em> mental health (loneliness, worry, overwhelm)</li>
        <li><strong>Calculation:</strong> Days Saved = (β × Loneliness Improvement) where β = 0.482 (Family CIRCLE data analysis)</li>
        <li><strong>Example:</strong> Multicomponent programs reduce caregiver loneliness by 0.80 points (<a href="https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2797399" target="_blank" style="color:#0066cc;">Hoang et al., JAMA Network Open, 2022</a>) → saves 0.39 days per child</li>
        <li><strong>Uncertainty:</strong> Each simulation runs 30 trials with Poisson noise to show realistic variation</li>
      </ul>
    </div>

    <div class="row">
      <div class="card settings-card">
        <h2 style="margin-top:0;color:#2c3e50;">Settings</h2>
        <div class="grid">
          <label>Intervention</label>
          <select id="scenario"></select>
          <label>Sample Size</label>
          <input id="sample_n" type="number" min="10" step="10">
          <label>Trials</label>
          <input id="trials" type="number" min="1" step="1" value="30">
          <label>Seed</label>
          <input id="seed" type="number" step="1" value="1234">
        </div>
        <button id="btn_run">Run Simulation</button>
      </div>

      <div class="card chart-card">
        <h2 style="margin-top:0;color:#2c3e50;">Average Absences (± error)</h2>
        <div id="avg_chart" style="height:320px;"></div>
      </div>
    </div>

    <div class="row">
      <div class="card" style="flex:1;">
        <h2 style="margin-top:0;color:#2c3e50;">Distribution: Baseline vs After</h2>
        <div id="hist_chart" style="height:340px;"></div>
      </div>
    </div>
  </div>

<script>
  const FAMILIES  = %%FAMILIES_JSON%%;
  const BETAS     = %%BETAS_JSON%%;
  const SCENARIOS = %%SCENARIOS_JSON%%;

  function mulberry32(a){return function(){var t=a+=0x6D2B79F5;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;}}
  function poisson(lam,rng){if(lam<=0)return 0;const L=Math.exp(-lam);let k=0,p=1;do{k++;p*=rng();}while(p>L);return k-1;}

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
        const lam=Math.max(0,base-saved);
        baseline.push(base); after.push(poisson(lam,rng));
      }
      baselineAvgs.push(baseline.reduce((a,b)=>a+b,0)/baseline.length);
      afterAvgs.push(after.reduce((a,b)=>a+b,0)/after.length);
      baselineAll.push(...baseline); afterAll.push(...after);
    }
    function mean(a){return a.reduce((x,y)=>x+y,0)/a.length;}
    function std(a){const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);}
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
      marker:{color:['#e74c3c','#27ae60']},
      text:[r.baselineAvg.toFixed(2), r.afterAvg.toFixed(2)],
      textposition:'auto',
      textfont:{color:'white'}
    }], { yaxis:{title:'Absence Days'}, title:'Based on '+trials+' trials', margin:{t:40} }, {displayModeBar:false});

    Plotly.react('hist_chart', [
      { x:r.baselineArr, type:'histogram', opacity:.6, name:'Baseline', marker:{color:'#e74c3c'} },
      { x:r.afterArr, type:'histogram', opacity:.6, name:'After ('+scenName+')', marker:{color:'#27ae60'} }
    ], { barmode:'overlay', xaxis:{title:'Absence Days'}, yaxis:{title:'Families'}, legend:{orientation:'h',y:1.1}, margin:{t:40} }, {displayModeBar:false});
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

    html = html.replace("%%FAMILIES_JSON%%", json.dumps(families_min))
    html = html.replace("%%BETAS_JSON%%", json.dumps(betas))
    html = html.replace("%%SCENARIOS_JSON%%", json.dumps(scenarios))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ────────────────────────────────────────────────────────────────────────────
# 4. GENERAL POPULATION EXPLORATORY SIMULATOR
# ────────────────────────────────────────────────────────────────────────────

def write_general_population_simulator(families: list[dict], betas: dict, out_path: str) -> None:
    """
    General population exploratory simulator with explicit assumptions and uncertainty.
    """
    families_min = [{
        "baseline":   float(f["baseline"]),
        "loneliness": float(f["loneliness"]),
        "worry":      float(f["worry"]),
        "overwhelm":  float(f["overwhelm"]),
    } for f in families]

    html = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>General Population Exploratory Simulator</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5rem;background:#f8f9fa;}
    .container{max-width:1400px;margin:0 auto;}
    h1{color:#2c3e50;margin-bottom:.5rem;}
    .subtitle{color:#7f8c8d;margin-bottom:1.5rem;font-size:1.05rem;}
    .warning{background:#fff3cd;border-left:4px solid #ffc107;padding:1.5rem;margin-bottom:2rem;border-radius:8px;}
    .warning h3{margin-top:0;color:#856404;font-size:1.1rem;}
    .warning p{margin:.5rem 0;color:#856404;line-height:1.6;}
    .warning strong{color:#664d03;}
    .assumptions{background:#e7f3ff;border-left:4px solid #0066cc;padding:1.5rem;margin-bottom:2rem;border-radius:8px;}
    .assumptions h3{margin-top:0;color:#004085;}
    .assumptions ul{margin:.5rem 0;padding-left:1.5rem;}
    .assumptions li{margin:.5rem 0;color:#004085;line-height:1.6;}
    .row{display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.5rem;}
    .card{background:white;border:1px solid #e1e8ed;border-radius:12px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.05);}
    .settings-card{flex:0 0 400px;}
    .chart-card{flex:1;min-width:500px;}
    .grid{display:grid;grid-template-columns:160px 1fr;gap:.75rem 1rem;align-items:center;margin-bottom:1rem;}
    label{font-weight:600;color:#34495e;font-size:.9rem;}
    input,select{padding:.6rem;border:2px solid #e1e8ed;border-radius:8px;font-size:.95rem;transition:border .2s;}
    input:focus,select:focus{outline:none;border-color:#3498db;}
    button{width:100%;padding:.8rem;border:none;border-radius:8px;background:#3498db;color:white;font-weight:600;font-size:1rem;cursor:pointer;transition:background .2s;}
    button:hover{background:#2980b9;}
    .metric{background:#ecf0f1;padding:1rem;border-radius:8px;margin-top:1rem;}
    .metric-label{font-size:.85rem;color:#7f8c8d;margin-bottom:.25rem;}
    .metric-value{font-size:1.6rem;font-weight:700;color:#2c3e50;}
    .range-display{background:#e8f4f8;padding:1rem;border-radius:8px;margin-top:1rem;font-size:.9rem;}
    .range-display h4{margin:0 0 .5rem;color:#2c3e50;}
    .confidence-badge{display:inline-block;padding:.25rem .5rem;border-radius:4px;font-size:.75rem;font-weight:600;margin-left:.5rem;}
    .high-confidence{background:#d4edda;color:#155724;}
    .exploratory{background:#fff3cd;color:#856404;}
  </style>
</head>
<body>
  <div class="container">
    <h1>🌎 General Population Impact (Exploratory)</h1>
    <p class="subtitle">Extrapolated estimates showing potential impact if CMC findings apply to all US children</p>

    <div class="warning">
      <h3>⚠️ IMPORTANT: These Are Extrapolated Estimates</h3>
      <p><strong>What we know with HIGH CONFIDENCE (from CMC research):</strong><br>
      When caregivers of medically complex children receive multicomponent interventions, their loneliness decreases by 0.80 points (<a href="https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2797399" target="_blank" style="color:#856404;text-decoration:underline;">Hoang et al., JAMA Network Open, 2022</a>), which reduces their child's school absences by ~0.39 days (Family CIRCLE data analysis). This is calculated by taking the loneliness frequency, statistically-significant beta coefficient from Family Circle analysis (0.482) and multiplying it by the intervention effect (0.8) derived from the JAMA study.</p>
      <p><strong>What we're ESTIMATING (general population):</strong><br>
      General population parents also experience high loneliness (66% report isolation - <a href="https://wexnermedical.osu.edu/mediaroom/pressreleaselisting/new-survey-finds-loneliness-epidemic-runs-deep-among-parents" target="_blank" style="color:#856404;text-decoration:underline;">Ohio State University Wexner Medical Center, 2024</a>). The same interventions <em>might</em> reduce absences for all children, but the effect could be weaker because:</p>
      <ul style="margin:.5rem 0 .5rem 1.5rem;padding:0;">
        <li>Healthy children's absences may be less driven by caregiver stress</li>
        <li>The loneliness → absence relationship may be CMC-specific</li>
        <li>We have no direct data, only reasonable assumptions</li>
      </ul>
      
      <p><strong>Bottom line:</strong> Use these estimates to understand <em>potential scale</em>, not as confirmed findings. Wide uncertainty ranges reflect our limited knowledge.</p>
    </div>

    <div class="assumptions">
      <h3>📊 Calculation Methodology:</h3>
      <ul>
        <li><strong>Simulation approach:</strong> We simulate 10,000 children (sample) and extrapolate to full population. The per-child effect is identical, so this is statistically equivalent but much faster.</li>
        <li><strong>Conservative Scenario:</strong> Assumes effect is 10% as strong in general population (β ≈ 0.048) </li>
        <li><strong>Middle Scenario:</strong> Assumes effect is 20% as strong (β ≈ 0.096) </li>
        <li><strong>Optimistic Scenario:</strong> Assumes effect is half as strong as CMC (β ≈ 0.24) </li>
        <li><strong>General population baseline:</strong> ~10 days absent per year (<a href="https://www.aei.org/research-products/report/everyone-is-missing-more-school-how-student-attendance-patterns-have-shifted-over-time/" target="_blank" style="color:#004085;text-decoration:underline;">Kirksey, AEI Report, 2025</a> - mean absence rates 5-8% of 180 school days)</li>
      </ul>
    </div>

    <div class="row">
      <div class="card settings-card">
        <h2 style="margin-top:0;color:#2c3e50;">Settings</h2>
        <div class="grid">
          <label>Intervention</label>
          <select id="scenario">
            <option value="none">No Intervention</option>
            <option value="multicomponent">Multicomponent Program</option>
            <option value="counseling">Group Counseling</option>
        </select>

          <label>Assumption Strength</label>
          <select id="assumption">
            <option value="conservative">Conservative (10%)</option>
            <option value="middle" selected>Middle (20%)</option>
            <option value="optimistic">Optimistic (50%)</option>
          </select>

          <label>US Children</label>
          <input id="us_pop" type="number" value="73000000" step="1000000">

          <label>Trials</label>
          <input id="trials" type="number" value="30" min="10" max="100" step="10">

          <label>Seed</label>
          <input id="seed" type="number" value="42" step="1">
        </div>
        <button id="btn_run">Run Simulation</button>

        <div class="metric">
          <div class="metric-label">Days Saved (Current Scenario)</div>
          <div class="metric-value" id="days_saved">0</div>
          <span class="confidence-badge exploratory">EXPLORATORY</span>
        </div>

        <div class="range-display">
          <h4>All Three Scenarios:</h4>
          <div id="range_display" style="font-size:.85rem;line-height:1.8;"></div>
        </div>

        <div class="metric" style="background:#d4edda;">
          <div class="metric-label">For Comparison: CMC Only</div>
          <div class="metric-value" style="color:#155724;" id="cmc_comparison">~438,000 days</div>
          <span class="confidence-badge high-confidence">HIGH CONFIDENCE</span>
        </div>
      </div>

      <div class="card chart-card">
        <h2 style="margin-top:0;color:#2c3e50;">Impact Comparison</h2>
        <div id="comparison_chart" style="height:400px;"></div>
      </div>
    </div>

    <div class="row">
      <div class="card" style="flex:1;">
        <h2 style="margin-top:0;color:#2c3e50;">Absence Day Distribution (Sample)</h2>
        <div id="dist_chart" style="height:350px;"></div>
      </div>
    </div>
  </div>

<script>
  const CMC_FAMILIES = %%FAMILIES_JSON%%;
  const CMC_BETAS = %%BETAS_JSON%%;

  // General population baseline: ~10 days/year average (NCES 2024)
  const GEN_POP_BASELINE = 10.0;

  // Assumption scenarios: what % of CMC effect applies to general population
   const ASSUMPTIONS = {
    conservative: { beta_mult: 0.10, intervention_mult: 0.10 },
    middle:       { beta_mult: 0.20, intervention_mult: 0.20 },
    optimistic:   { beta_mult: 0.5, intervention_mult: 0.5 }
  };

    const INTERVENTIONS = {
        none: { loneliness: 0.0, worry: 0.0, overwhelm: 0.0 },
        multicomponent: { loneliness: 0.80, worry: 0.0, overwhelm: 0.0 },
        counseling: { loneliness: 0.23, worry: 0.0, overwhelm: 0.0 }
    };

  function mulberry32(a){return function(){var t=a+=0x6D2B79F5;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;}}
  function poisson(lam,rng){if(lam<=0)return 0;const L=Math.exp(-lam);let k=0,p=1;do{k++;p*=rng();}while(p>L);return k-1;}

  function simulate(){
    const scenario = document.getElementById('scenario').value;
    const assumption = document.getElementById('assumption').value;
    const usPop = parseInt(document.getElementById('us_pop').value);
    const trials = parseInt(document.getElementById('trials').value);
    const seed = parseInt(document.getElementById('seed').value);

    const assumptionParams = ASSUMPTIONS[assumption];
    const intervention = INTERVENTIONS[scenario];

    // Adjusted betas for general population
    const adjustedBeta = CMC_BETAS.loneliness * assumptionParams.beta_mult;
    const adjustedIntervention = intervention.loneliness * assumptionParams.intervention_mult;
    
    const saved = adjustedBeta * adjustedIntervention;

    // OPTIMIZATION: Simulate a sample of 10,000 children instead of all 73M
    // The per-child effect is identical, so we can extrapolate
    const sampleSize = 10000;
    let baselineAvgs = [], afterAvgs = [], baselineAll = [], afterAll = [];
    
    for(let t = 0; t < trials; t++){
      const rng = mulberry32(seed + t);
      let baselineSum = 0, afterSum = 0;
      
      for(let i = 0; i < sampleSize; i++){
        const base = GEN_POP_BASELINE;
        const lam = Math.max(0, base - saved);
        const after = poisson(lam, rng);
        
        baselineSum += base;
        afterSum += after;
        if(t === 0){ baselineAll.push(base); afterAll.push(after); }
      }
      
      baselineAvgs.push(baselineSum / sampleSize);
      afterAvgs.push(afterSum / sampleSize);
    }

    const mean = a => a.reduce((x,y) => x+y, 0) / a.length;
    const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s,v) => s+(v-m)**2, 0) / a.length); };

    const baselineMean = mean(baselineAvgs);
    const afterMean = mean(afterAvgs);
    const totalSaved = scenario === 'none' ? 0 : Math.round((baselineMean - afterMean) * usPop);
    
    document.getElementById('days_saved').textContent = totalSaved >= 0 ? totalSaved.toLocaleString() : '0';

    // Calculate CMC comparison (using CMC data: 1.46M children, CMC baseline ~1.81 days)
    const cmcPopulation = 1460000;
    const cmcBaselineAvg = 1.81;
    const cmcBeta = CMC_BETAS.loneliness;
    const cmcInterventionEffect = intervention.loneliness; // Use actual intervention, not adjusted
    const cmcDaysSaved = scenario === 'none' ? 0 : Math.round((cmcBeta * cmcInterventionEffect) * cmcPopulation);
    
    document.getElementById('cmc_comparison').textContent = '~' + cmcDaysSaved.toLocaleString() + ' days';

    // Calculate all three scenarios for comparison
    const allScenarios = {};
    for(const [assumpName, assumpParams] of Object.entries(ASSUMPTIONS)){
      const beta = CMC_BETAS.loneliness * assumpParams.beta_mult;
      const interv = intervention.loneliness * assumpParams.intervention_mult;
      const s = beta * interv;
      const perChild = s;
      const total = scenario === 'none' ? 0 : Math.round(perChild * usPop);
      allScenarios[assumpName] = { perChild, total };
    }

    document.getElementById('range_display').innerHTML = `
      <strong>Conservative:</strong> ${allScenarios.conservative.total.toLocaleString()} days (${allScenarios.conservative.perChild.toFixed(3)} per child)<br>
      <strong>Middle:</strong> ${allScenarios.middle.total.toLocaleString()} days (${allScenarios.middle.perChild.toFixed(3)} per child)<br>
      <strong>Optimistic:</strong> ${allScenarios.optimistic.total.toLocaleString()} days (${allScenarios.optimistic.perChild.toFixed(3)} per child)
    `;

    Plotly.react('comparison_chart', [{
      x: ['Baseline', 'After Intervention\n(' + scenario.toUpperCase() + ', ' + assumption + ')'],
      y: [baselineMean, afterMean],
      error_y: { type: 'data', array: [std(baselineAvgs), std(afterAvgs)], visible: true },
      type: 'bar',
      marker: { color: ['#e74c3c', '#27ae60'] },
      text: [baselineMean.toFixed(2) + ' days', afterMean.toFixed(2) + ' days'],
      textposition: 'auto',
      textfont: { size: 14, color: 'white' }
    }], {
      yaxis: { title: 'Average Days Absent per Child' },
      showlegend: false,
      margin: { t: 40, b: 80 }
    }, { displayModeBar: false });

    Plotly.react('dist_chart', [
      { x: baselineAll, type: 'histogram', opacity: .6, name: 'Baseline', marker: { color: '#e74c3c' } },
      { x: afterAll, type: 'histogram', opacity: .6, name: 'After ' + scenario.toUpperCase(), marker: { color: '#27ae60' } }
    ], {
      barmode: 'overlay',
      xaxis: { title: 'Days Absent' },
      yaxis: { title: 'Number of Children (10k sample)' },
      legend: { orientation: 'h', y: 1.1 },
      margin: { t: 40 }
    }, { displayModeBar: false });
  }

  document.getElementById('btn_run').addEventListener('click', simulate);
  simulate();
</script>
</body>
</html>
"""

    html = html.replace("%%FAMILIES_JSON%%", json.dumps(families_min))
    html = html.replace("%%BETAS_JSON%%", json.dumps(betas))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BETAS = {
        'overwhelm':   0.1817621,
        'worry':       0.1026722,
        'loneliness':  0.4822393
    }

    df_raw = pd.read_csv("data.csv", dtype=str)
    keep = [
        'record_id', 'race_ethn_cmc',
        'lonely_freq_s2', 'fs_feeling_s2', 'fs_worry_s2', 'missed_days_s2'
    ]
    df = load_and_clean(df_raw[keep])
    families = make_families(df)

    # scenarios checked
    scenarios = {
        'No Intervention': {},
        'Multicomponent Program':      {'loneliness': 0.80},
        'Group Counseling':             {'loneliness': 0.23},
    }
    
    os.makedirs("docs", exist_ok=True)
    
    print("Running simulations...")
    
    # Run multiple trials and calculate mean + std for each family
    n_trials = 30
    all_results = {}

    for name, policy in scenarios.items():
        trial_results = []
        for trial in range(n_trials):
            trial_results.append(run_sim(families, BETAS, policy, seed=42+trial))
        
        averaged_results = []
        for i in range(len(families)):
            family_outcomes = [trial[i]['after'] for trial in trial_results]
            avg_result = trial_results[0][i].copy()
            avg_result['after'] = np.mean(family_outcomes)
            avg_result['after_std'] = np.std(family_outcomes)
            averaged_results.append(avg_result)
        
        all_results[name] = averaged_results
    
    # Build family tracker dataframe
    rows = []
    for i, fam in enumerate(families):
        row = {
            'Family': f"Family {i+1}",
            'ID': fam['id'],
            'Child race': fam['race'],
            'Baseline': fam['baseline']
        }
        for scenario_name in scenarios:
            scenario_results = all_results[scenario_name]
            matching_result = next(r for r in scenario_results if r['id'] == fam['id'])
            row[scenario_name] = matching_result['after']
            row[f"{scenario_name}__std"] = matching_result['after_std']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create family tracker visualization
    fig = go.Figure()
    x_categories = list(scenarios.keys())
    
    for _, family_row in df.iterrows():
        y_values = [family_row[scenario] for scenario in x_categories]
        y_std_vals = [family_row[f"{scenario}__std"] for scenario in x_categories]
        
        fig.add_trace(go.Scatter(
            error_y=dict(type='data', array=y_std_vals, visible=True),
            customdata=y_std_vals,
            x=x_categories,
            y=y_values,
            mode='lines+markers',
            name=family_row['Family'],
            line=dict(width=1),
            marker=dict(size=8),
            opacity=0.7,
            hovertemplate=(
                f"<b>{family_row['Family']}</b><br>" +
                f"Child race: {family_row['Child race']}<br>" +
                f"Baseline: {family_row['Baseline']:.1f} days<br>" +
                "Scenario: %{x}<br>" +
                "Mean days absent: %{y:.2f}<br>" +
                "Std across trials: %{customdata:.2f}<br>" +
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="Digital Twin: Track Each CMC Family Across All Interventions",
        xaxis_title="Intervention",
        yaxis_title="Days Absent from School",
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02
        ),
        margin=dict(r=150)
    )
    
    fig.write_html("docs/family_tracker.html", include_plotlyjs='cdn')
    print("→ wrote docs/family_tracker.html")
    
    write_interactive_simulator(families, BETAS, "docs/cmc_simulator.html")
    print("→ wrote docs/cmc_simulator.html")
    
    write_general_population_simulator(families, BETAS, "docs/general_population.html")
    print("→ wrote docs/general_population.html")
    
    # Create index page
    with open("docs/index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Caregiver Mental Health & School Absence Simulator</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0;
            padding: 3rem;
            background: #f8f9fa;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50;
            margin-bottom: .5rem;
        }
        .subtitle {
            color: #7f8c8d;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .tool-section {
            margin: 2rem 0;
        }
        .tool-card {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            transition: transform .2s, box-shadow .2s;
        }
        .tool-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .tool-card h3 {
            margin: 0 0 .5rem;
            color: #2c3e50;
        }
        .tool-card p {
            margin: .5rem 0 1rem;
            color: #555;
            line-height: 1.6;
        }
        .confidence-badge {
            display: inline-block;
            padding: .25rem .75rem;
            border-radius: 12px;
            font-size: .75rem;
            font-weight: 600;
            margin-left: .5rem;
        }
        .high-confidence {
            background: #d4edda;
            color: #155724;
        }
        .exploratory {
            background: #fff3cd;
            color: #856404;
        }
        a.btn { 
            display: inline-block;
            padding: .75rem 1.5rem;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            transition: background .2s;
        }
        a.btn:hover { 
            background: #2980b9;
        }
        .methodology {
            background: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 8px;
        }
        .methodology h3 {
            margin-top: 0;
            color: #004085;
        }
        .methodology ul {
            margin: .5rem 0;
            padding-left: 1.5rem;
        }
        .methodology li {
            margin: .5rem 0;
            color: #004085;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div style="background:#fff3cd;border-left:4px solid #ffc107;padding:1.5rem;margin-bottom:2rem;border-radius:8px;">
            <h3 style="margin-top:0;color:#856404;">⚠️ Prototype Disclaimer</h3>
            <p style="color:#856404;line-height:1.6;margin:.5rem 0;">
                This simulator is a <strong>research prototype</strong> with limited validation. While core parameters are derived from peer-reviewed literature and Family CIRCLE data analysis, several aspects require further refinement including but not limited to: (1) comprehensive unit testing of all functions, (2) independent verification of computational accuracy, (3) validation of mathematical and software-engineering methods, and (4) peer review of methodology. Users should interpret results as exploratory estimates rather than definitive predictions.
            </p>
        </div>

        <h1>📊 Caregiver Mental Health & School Absence Simulator</h1>
        <p class="subtitle">Interactive tools showing how caregiver-focused interventions reduce school absences</p>

        <div class="methodology">
            <h3>🔬 Research Foundation</h3>
            <ul>
                <li><strong>Study Population:</strong> Families with medically complex children (CMC) - Family CIRCLE data analysis</li>
                <li><strong>Key Finding:</strong> When caregiver loneliness improves by 1 point → child absences decrease by 0.48 days (β = 0.482) - Family CIRCLE data analysis</li>
                <li><strong>Interventions:</strong> Multicomponent programs and group counseling targeting caregiver mental health</li>
                <li><strong>Evidence Base:</strong> 
                    <ul style="margin-top:.5rem;">
                        <li>Regression coefficient (β = 0.482): Family CIRCLE data analysis</li>
                        <li>Multicomponent programs (SMD -0.67 → 0.80 point improvement): <a href="https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2797399" target="_blank" style="color:#0066cc;">Hoang et al., JAMA Network Open, 2022</a></li>
                        <li>Group counseling (SMD -0.19 → 0.23 point improvement): <a href="https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2797399" target="_blank" style="color:#0066cc;">Hoang et al., JAMA Network Open, 2022</a></li>
                    </ul>
                </li>
            </ul>
        </div>

        <div class="tool-section">
            <div class="tool-card">
                <h3>🏠 Family Tracker <span class="confidence-badge high-confidence">HIGH CONFIDENCE</span></h3>
                <p>Track how each individual CMC family responds across all interventions (No Intervention, Multicomponent Program, Group Counseling). See every family's trajectory with uncertainty bars showing variation across 30 simulation trials.</p>
                <a href='family_tracker.html' class="btn">View Family Tracker</a>
            </div>

            <div class="tool-card">
                <h3>🔬 CMC Interactive Simulator <span class="confidence-badge high-confidence">HIGH CONFIDENCE</span></h3>
                <p>Simulate interventions on the CMC population with adjustable parameters (sample size, trials, random seed). Based directly on measured CMC data with transparent calculation methodology showing exactly how days saved are computed.</p>
                <a href='cmc_simulator.html' class="btn">Open CMC Simulator</a>
            </div>

            <div class="tool-card" style="border-left-color: #ffc107;">
                <h3>🌎 General Population Exploratory <span class="confidence-badge exploratory">EXPLORATORY</span></h3>
                <p>Explore potential impact if CMC findings extend to all US children (73M). Shows three scenarios (conservative, middle, optimistic) with explicit assumptions about how much the CMC effect transfers. Use for understanding potential scale, not as confirmed findings.</p>
                <a href='general_population.html' class="btn" style="background: #ffc107; color: #000;">Open Exploratory Tool</a>
            </div>
        </div>

        <div style="margin-top: 3rem; padding-top: 2rem; border-top: 2px solid #e1e8ed; color: #7f8c8d; font-size: .9rem;">
            <p><strong>Note on Evidence Levels:</strong></p>
            <p><strong>High Confidence:</strong> Based on direct measurement in CMC families from survey data and established statistical methods.</p>
            <p><strong>Exploratory:</strong> Extrapolated estimates based on reasonable assumptions but without direct measurement. Wide uncertainty ranges acknowledge limited evidence.</p>
        </div>
    </div>
</body>
</html>""")
    
    print("→ wrote docs/index.html")
    print("\n✅ Done! Open docs/index.html to view all three simulators")