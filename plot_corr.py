import os
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from config import *  # Import configuration from config.py
from matplotlib.ticker import MaxNLocator

# Set global style parameters first
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 22,               # Default font size
    'axes.titlesize': 22,          # Title font size
    'axes.labelsize': 22,          # Axis label font size
    'xtick.labelsize': 22,         # X-axis tick label size
    'ytick.labelsize': 22,         # Y-axis tick label size
    'legend.fontsize': 12,         # Legend font size
    'lines.linewidth': 3,          # Line width
    'lines.markersize': 10,        # Marker size
    'grid.alpha': 0.4,             # Grid transparency
    'figure.autolayout': True,     # Enable auto layout
    'figure.dpi': 300,             # High resolution output
    'savefig.dpi': 300,            # Save figure DPI
    'savefig.bbox': 'tight'        # Tight bounding box
})

colors = { "PatchTST": "C5", "NHITS": "C6", "DeepAR": "C7", "Informer": "C3", "FEDformer": "C9" }

def calculate_accuracy(preds, trues, eps):
    """Safe accuracy calculation with validation checks"""
    # Convert to numeric arrays, forcing invalid to NaN
    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    
    # Validate inputs
    if preds.shape != trues.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs trues {trues.shape}")
    
    # Calculate absolute errors
    with np.errstate(all='raise'):
        try:
            errors = np.abs(preds - trues)
        except FloatingPointError:
            print("Floating point error in subtraction")
            return np.nan
    
    # Filter finite values
    finite_mask = np.isfinite(errors)
    valid_errors = errors[finite_mask]
    
    # Edge cases
    if len(valid_errors) == 0:
        print("Warning: No valid error values")
        return np.nan
    
    # Calculate accuracy
    accuracy = np.mean(valid_errors <= eps)
    
    # Final sanity check (should never trigger)
    if not (0 <= accuracy <= 1):
        print(f"Invalid accuracy {accuracy} with eps={eps}")
        print(f"Error stats - min: {valid_errors.min()}, max: {valid_errors.max()}")
        print(f"Pred stats - min: {preds.min()}, max: {preds.max()}")
        print(f"True stats - min: {trues.min()}, max: {trues.max()}")
        return np.nan
    
    return accuracy

# Directories setup
RESULTS_MODEL_DIR = "results/ResultsModels"
RESULTS_LYAP_DIR = "results/ResultsLyap"
RESULTS_PIMAX_DIR = "results/ResultsPimax"
PLOT_OUTPUT_DIR = "Plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

def clean_pimax_string(s):
    """Clean the pimax string by removing numpy float decorations"""
    return s.replace("np.float64", "").replace("np.float", "").replace("(", "").replace(")", "")

Lyap_exp = {}
for dataset_name, dataset_cfg in DATASETS_CONFIG.items():
    print(f"Processing dataset: {dataset_name}")
    H = dataset_cfg["horizon"]
    epsilons = dataset_cfg["epsilons"]
    horizons = list(range(1, H + 1))

    # ==================================================================
    # 1. Process model results and calculate accuracies
    # ==================================================================
    model_accuracies = {}
    pattern = os.path.join(RESULTS_MODEL_DIR, f"*_{dataset_name}.csv")
    
    for filepath in glob.glob(pattern):
        model_name = os.path.basename(filepath).split('_')[0]
        df_model = pd.read_csv(filepath)
        
        # Dictionary to store accuracies for this model
        model_accuracies[model_name] = {eps: [] for eps in epsilons}
        
        # Calculate accuracy for each horizon and epsilon
        for h in range(1, H + 1):
            pred_col = f'pred{h}'
            true_col = f'true{h}'
            
            if pred_col in df_model.columns and true_col in df_model.columns:                
                for eps in epsilons:
                    accuracy = calculate_accuracy(df_model[pred_col], df_model[true_col], eps)
                    model_accuracies[model_name][eps].append(accuracy)
            else:
                for eps in epsilons:
                    model_accuracies[model_name][eps].append(np.nan)

    # ==================================================================
    # 2. Process Lyapunov predictability bounds
    # ==================================================================
    lyap_path = os.path.join(RESULTS_LYAP_DIR, f"{dataset_name}_lyapunov_exponents.csv")
    
    
    if os.path.exists(lyap_path):
        df_lyap = pd.read_csv(lyap_path)
        # Assume columns are ordered by config's epsilon order
        lyap_min_pos = df_lyap["exponent"].values
        lyap_min_pos = min([l for l in lyap_min_pos if l > 0], default=0)
        Lyap_exp[dataset_name] = lyap_min_pos
    else:
        print(f"Warning: Lyapunov file not found - {lyap_path}")
        Lyap_exp[dataset_name] = np.nan

    # ==================================================================
    # 3. Process Pimax predictability bounds
    # ==================================================================
    pimax_path = os.path.join(RESULTS_PIMAX_DIR, f"{dataset_name}_pimax_h.csv")
    pimax_bounds = {eps: [] for eps in epsilons}
    lyap_bounds = {eps: [] for eps in epsilons}
    
    if os.path.exists(pimax_path):
        df_pimax = pd.read_csv(pimax_path)
        
        for _, row in df_pimax.iterrows():
            try:
                eps_val = float(row[0])
                if eps_val not in epsilons:
                    continue
                    
                # Clean and convert the bound string
                cleaned = clean_pimax_string(row[1])
                bound_list = ast.literal_eval(cleaned)[:H]  # Truncate to horizon length
                pimax_bounds[eps_val] = bound_list
                pimax_0 = bound_list[0]
                lyap_bounds[eps_val] = [pimax_0 * np.exp(-Lyap_exp[dataset_name] * (h-1)) for h in horizons]
            except (ValueError, SyntaxError) as e:
                print(f"Error processing Pimax bounds: {e}")
                continue
    else:
        print(f"Warning: Pimax file not found - {pimax_path}")
        for eps in epsilons:
            pimax_bounds[eps] = [np.nan] * H

    # ==================================================================
    # 4. Create and save plots for each epsilon
    # ==================================================================
    for epsilon in epsilons:
        plt.figure(figsize=(10, 8))

        print(f"Processing epsilon: {epsilon}, Pimax bounds: {pimax_bounds[epsilon]}")
        
        # Plot bounds with enhanced styling
        plt.plot(
            horizons,
            pimax_bounds[epsilon],
            'D-.',
            #label="Pimax Bound",
            markersize=10,
            markeredgewidth=2,
            markeredgecolor='black'
        )

        plt.plot(
            horizons,
            lyap_bounds[epsilon],
            's--',
            #label="Lyapunov Bound",
            markersize=10,
            markeredgewidth=2,
            markeredgecolor='black'
        )
        
        # Plot model accuracies with consistent styling
        for model_name, acc_data in model_accuracies.items():
            plt.plot(
                horizons,
                acc_data[epsilon],
                'o-',
                #label=f"{model_name}",
                color=colors[model_name],
                alpha=0.9,
                markersize=8,
                markeredgewidth=1,
                markeredgecolor='black'
            )
        
        # Enhanced title and labels
        plt.title(f"{dataset_name} ($\epsilon = {epsilon}$)", pad=20)  # pad adds space above title
        plt.xlabel("Horizon (h)")  # labelpad adds space below label
        plt.ylabel("Accuracy / Bound")
        
        # Axis configuration
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5, nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        
        # Enhanced legend
        ''''
        legend = plt.legend(
            frameon=True,
            framealpha=1,
            edgecolor='black',
            bbox_to_anchor=(1.05, 1),  # Moves legend outside plot
            loc='upper left'
        )
        '''
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Save plot
        plot_file = os.path.join(PLOT_OUTPUT_DIR, f"{dataset_name}_epsilon_{epsilon}.png")
        plt.savefig(plot_file, bbox_inches='tight', dpi=300, transparent=False)
        plt.close()
        print(f"Saved plot: {plot_file}")

print("Processing completed!")