import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()

def var_calc(var):
    scores = pd.to_numeric(df[var], errors='coerce').dropna()
    valid_scores = scores[scores.isin([1,2,3,4,5])].astype(int)
    if var == 'lonely_freq_s2':
        valid_scores = valid_scores.replace({1:5,2:4,3:3,4:2,5:1})
    return valid_scores

def plot_mean_absences(var, xlabel):
    # you get a whole column of valid scores for A specific var
    cleaned = var_calc(var)
    # slicing rows
    # so we get a set of rows. which rows? Those that correspond with valid_scores
    sub = df.loc[cleaned.index].copy()
    sub[var] = cleaned

    # ❌ THIS WAS MISSING: convert absences to numeric, drop invalid
    sub['missed_days_s2'] = pd.to_numeric(sub['missed_days_s2'], errors='coerce')
    # Drops any rows in your filtered DataFrame sub where the missed_days_s2 column is NaN
    sub = sub.dropna(subset=['missed_days_s2'])

    # basically, within each level, look at missed days and compute mean.
    # e.g. level -> within worry frequency, make a group for score 1 worries, score 2 worries, etc, and compute the mean
    # of absences within each group
    means = sub.groupby(var)['missed_days_s2'].mean()

    fig, ax = plt.subplots()
    ax.bar(means.index, means.values)
    ax.set_xticks(means.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Missed School Days (s2)")
    ax.set_title(f"Mean Missed School Days by {xlabel}")
    plt.tight_layout()
    plt.show()

# then your three calls:
plot_mean_absences('fs_feeling_s2',  'Overwhelm level due to financial stress \n(1=Very Bad … 5=Very Good)')
plot_mean_absences('fs_worry_s2',    'Worry due to financial stress Score\n(1=Always Worried … 5=Never Worried)')
plot_mean_absences('lonely_freq_s2', 'Loneliness Frequency\n(1=Never … 5=Always)')
