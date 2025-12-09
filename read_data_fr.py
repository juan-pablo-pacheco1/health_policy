import pandas as pd
import numpy as np
from simulation import _clamp_int_15, _reverse_1to10_higher_better, _map_1to10_to_1to5

# Load your CSV file
df = pd.read_csv('data.csv')  # Replace 'your_file.csv' with your actual file name

def var_calc(var):
    # Apply the same data cleaning transformations as in simulation.py
    
    # First convert to numeric, coerce bad entries to NaN
    raw_scores = pd.to_numeric(df[var], errors='coerce')
    total_entries = len(df[var])
    
    if var == 'lonely_freq_s2':
        # loneliness is already 1..5, higher=worse → clamp/round to be safe
        cleaned_scores = raw_scores.apply(_clamp_int_15)
        # Keep only valid scores (1-5)
        valid_scores = cleaned_scores.dropna()
        var_name = 'loneliness'
        
    elif var == 'fs_feeling_s2':
        # feeling is 1..10 where higher=better → flip to higher=worse, then map to 1..5
        cleaned_scores = raw_scores.apply(_reverse_1to10_higher_better).apply(_map_1to10_to_1to5)
        # Keep only valid scores (1-5)
        valid_scores =  cleaned_scores.dropna()
        var_name = 'overwhelm'
        
    elif var == 'fs_worry_s2':
        # worry is 1..10 where higher=better → flip to higher=worse, then map to 1..5
        cleaned_scores = raw_scores.apply(_reverse_1to10_higher_better).apply(_map_1to10_to_1to5)
        # Keep only valid scores (1-5)
        valid_scores = cleaned_scores.dropna()
        var_name = 'worry'
    
    else:
        # For other variables, just keep integers 1 through 5
        valid_scores = raw_scores[raw_scores.isin([1, 2, 3, 4, 5])]
        var_name = var

    # STEP 3: Compute statistics
    mean_score = valid_scores.mean()
    std_dev    = valid_scores.std()

    # STEP 4: Display results
    print(f"Frequency results for '{var}' (cleaned as '{var_name}')")
    print("--------------------------------------")
    print(f"Total entries:   {total_entries}")
    print(f"Valid responses: {len(valid_scores)}")
    print(f"Data retention:  {len(valid_scores)/total_entries*100:.2f}%")
    print(f"Mean score:      {mean_score:.2f}")
    print(f"Std. dev.:       {std_dev:.2f}")
    print(f"Scale note:      1-5 where higher = worse")
    print()

def main():
    print("Data cleaning applied as per simulation.py:")
    print("- lonely_freq_s2: Already 1-5 (higher=worse), clamped to valid range")
    print("- fs_feeling_s2: 1-10 (higher=better) → flipped → mapped to 1-5 (higher=worse)")  
    print("- fs_worry_s2: 1-10 (higher=better) → flipped → mapped to 1-5 (higher=worse)")
    print("=" * 60)
    print()
    
    var_calc('fs_feeling_s2')
    var_calc('fs_worry_s2') 
    var_calc('lonely_freq_s2')

if __name__ == "__main__":
    main()