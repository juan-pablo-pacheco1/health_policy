import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─── 1) Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# ─── 2) Outcome: raw → numeric → midpoint ────────────────────────────────────
df["missed_days_s2_num"] = pd.to_numeric(df["missed_days_s2"], errors="coerce")
midpoint_map = {0:0,1:2,2:5,3:8.5,4:11}
df["school_absences"] = df["missed_days_s2_num"].map(midpoint_map)
# If you prefer the raw 0–4, uncomment:
# df["school_absences"] = df["missed_days_s2_num"]

# ─── 3) Exposures ─────────────────────────────────────────────────────────────
df["overwhelm_freq"]  = pd.to_numeric(df["fs_feeling_s2"], errors="coerce")
df["worry_freq"]      = pd.to_numeric(df["fs_worry_s2"],   errors="coerce")
df["loneliness_freq"] = pd.to_numeric(df["lonely_freq_s2"],errors="coerce")

# ─── 4) Continuous ages ───────────────────────────────────────────────────────
df["caregiver_age"] = pd.to_numeric(df["age_yrs_s1"],     errors="coerce")
df["child_age"]     = pd.to_numeric(df["age_yrs_cmc_s1"], errors="coerce")

# ─── 5) Sex binaries ──────────────────────────────────────────────────────────
df["child_sex"]     = pd.to_numeric(df["bio_sex_birth_cmc_s1"],errors="coerce")
df.loc[~df["child_sex"].isin([0,1]), "child_sex"] = np.nan
df["caregiver_sex"] = pd.to_numeric(df["bio_sex_birth_2_s1"],  errors="coerce")
df.loc[~df["caregiver_sex"].isin([0,1]), "caregiver_sex"] = np.nan

# ─── 6) Full race/ethnicity categories ───────────────────────────────────────
race_map = {
  "1":"American Indian/Alaska Native","2":"Black/African American",
  "3":"Asian","4":"Native Hawaiian/Other Pacific Islander","5":"White",
  "6":"Hispanic/Latino","7":"Multiracial","8":"Some other race",
  ".m": "Prefer not/missing",".l":"Prefer not/missing",".d":"Prefer not/missing"
}
for who,var in [("child","race_ethn_cmc"),("caregiver","race_ethn_caregiver")]:
    cat = df[var].astype(str).map(race_map).fillna("Prefer not/missing")
    df[f"{who}_race_cat"] = cat

race_child     = pd.get_dummies(df["child_race_cat"],     prefix="child_race")
race_child.drop(columns=["child_race_White"], inplace=True)
race_caregiver = pd.get_dummies(df["caregiver_race_cat"], prefix="caregiver_race")
race_caregiver.drop(columns=["caregiver_race_White"], inplace=True)

# ─── 7) SES flags ────────────────────────────────────────────────────────────
df["edu_college_up"]    = (pd.to_numeric(df["edu_cat"], errors="coerce") >= 3).astype(int)
df["married_partnered"] = (df["marital_status_s1"] == 1).astype(int)
df["income_less35k"]    = df["family_income_s1"].isin([1,2,3,4]).astype(int)

# ─── 8) Assemble regression DataFrame ───────────────────────────────────────
df = pd.concat([df, race_child, race_caregiver], axis=1)
model_vars = (
    ["school_absences",
     "overwhelm_freq","worry_freq","loneliness_freq",
     "child_age","child_sex"] + list(race_child.columns) +
    ["caregiver_age","caregiver_sex"] + list(race_caregiver.columns) +
    ["edu_college_up","married_partnered","income_less35k"]
)
sub = df[model_vars]

# ─── 9) Drop any missing ─────────────────────────────────────────────────────
print("Rows before dropna():", len(sub))
print(sub.isna().sum(), "\n")
reg_df = sub.dropna()
print("Rows after dropna():", len(reg_df), "\n")
if reg_df.empty:
    raise RuntimeError("No rows left after dropna(); check your mappings!")

# ─── 10) Split X/y and coerce to float ───────────────────────────────────────
y = reg_df["school_absences"].astype(float)
X = reg_df.drop(columns="school_absences").astype(float)

# ─── 11) Standardize all continuous predictors ──────────────────────────────
cont = ["overwhelm_freq","worry_freq","loneliness_freq","child_age","caregiver_age"]
X[cont] = (X[cont] - X[cont].mean()) / X[cont].std()

# ─── 12) Drop constant cols ─────────────────────────────────────────────────
consts = [c for c in X if X[c].std()==0]
print("Dropping constant columns:", consts, "\n")
X.drop(columns=consts, inplace=True)

# ─── 13) Add intercept and check & drop high‐VIF cols ────────────────────────
X = sm.add_constant(X, has_constant="add")
def vif_df(X):
    return pd.DataFrame([
        (col, variance_inflation_factor(X.values, i))
        for i,col in enumerate(X.columns) if col!="const"
    ], columns=["variable","VIF"])

v = vif_df(X)
print("Initial VIFs:\n", v, "\n")
while v["VIF"].max() > 10:
    drop = v.sort_values("VIF", ascending=False)["variable"].iloc[0]
    print(f"Dropping {drop} (VIF={v['VIF'].max():.1f})")
    X.drop(columns=[drop], inplace=True)
    v = vif_df(X)
print("Final VIFs:\n", v, "\n")

# ─── 14) Fit OLS & show condition number ─────────────────────────────────────
model = sm.OLS(y, X).fit()
print("Condition Number:", np.linalg.cond(X), "\n")
print(model.summary())

# ─── 15) Extract Table 5 ────────────────────────────────────────────────────
tbl = pd.DataFrame({
    "Estimate": model.params,
    "SE":        model.bse,
    "CI_lower":  model.conf_int().iloc[:,0],
    "CI_upper":  model.conf_int().iloc[:,1],
    "p‑value":   model.pvalues
})
tbl.index.name = "Variable"
print("\n— Table 5: School Absences (Overwhelm, Worry, Loneliness) —\n")
print(tbl.to_markdown(floatfmt=".3f"))
