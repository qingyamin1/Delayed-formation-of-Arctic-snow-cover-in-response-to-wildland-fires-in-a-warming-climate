

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from matplotlib.offsetbox import AnchoredText

# ---------------- CONFIG ---------------- #
FIGSIZE = (8.2, 3)
PANEL_FONT = 12
AXIS_FONT = 6
BAR_WIDTH = 0.22
CUSTOM_ORDER = ['Albedo', 'T2', 'LST', 'VPD', 'lat', 
                'shortwave', 'longwave', 'Def', 'BA', 'lon']

# Paths to models & data
PATHS = {
    "pre_model":    "Figure-3-data/XG_pre8*8_best_model.joblib",
    "pre_test":     "Figure-3-data/regridded_xgboost-merged_8*8_occurrence_features-pre-summer.csv",
    "during_model": "Figure-3-data/XG_during8*8_best_model.joblib",
    "during_test":  "Figure-3-data/regridded_xgboost-merged_8*8_occurrence_features-during-summer.csv",
    "post_model":   "Figure-3-data/XG_post8*8_best_model.joblib",
    "post_test":    "Figure-3-data/regridded_xgboost-merged_8*8_occurrence_features-post-summer.csv",
    "post_plot_vals": "Figure-3-data/XG_post8_best_model_plot_values.csv"
}



def compute_shap_values(model_path, test_path):
    """Load model, compute mean absolute SHAP values."""
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_path, index_col=0)
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_test)
    if isinstance(shap_val, list) or shap_val.ndim == 3:
        shap_val = shap_val[1] if len(shap_val) > 1 else shap_val[0]
    mean_abs_shap = np.abs(shap_val).mean(axis=0)
    return pd.DataFrame({'feature': X_test.columns, 'mean_abs_shap': mean_abs_shap}), X_test, shap_val

def plot_shap_bar(ax, df_pre, df_during, df_post):
    """Grouped bar chart for SHAP importance with safe merging."""
    # Normalize feature names
    for df in [df_pre, df_during, df_post]:
        df['feature'] = df['feature'].str.strip()
    
    # Outer merge to keep all features from all periods
    merged = (
        df_pre.merge(df_during, on='feature', suffixes=('_pre', '_during'), how='outer')
              .merge(df_post, on='feature', how='outer')
              .rename(columns={'mean_abs_shap': 'mean_abs_shap_post'})
    )
    
    # Fill any missing SHAP values with zero
    for col in ['mean_abs_shap_pre', 'mean_abs_shap_during', 'mean_abs_shap_post']:
        if col not in merged:
            merged[col] = 0  # Safety if column was missing after merge
    merged[['mean_abs_shap_pre', 'mean_abs_shap_during', 'mean_abs_shap_post']] = \
        merged[['mean_abs_shap_pre', 'mean_abs_shap_during', 'mean_abs_shap_post']].fillna(0)
    
    # Filter and reorder by CUSTOM_ORDER, keep only those present
    available_features = [f for f in CUSTOM_ORDER if f in merged['feature'].values]
    if not available_features:
        print("Warning: None of the CUSTOM_ORDER features found in SHAP data.")
    merged = merged.set_index('feature').loc[available_features].reset_index()
    
    # Plot bars
    x = np.arange(len(merged))
    ax.bar(x - BAR_WIDTH, merged['mean_abs_shap_pre'], width=BAR_WIDTH, label='Pre-fire', color='#ffb703')
    ax.bar(x, merged['mean_abs_shap_during'], width=BAR_WIDTH, label='During-fire', color='#219ebc')
    ax.bar(x + BAR_WIDTH, merged['mean_abs_shap_post'], width=BAR_WIDTH, label='Post-fire', color='#d62828')
    
    ax.set_xticks(x)
    ax.set_xticklabels(merged['feature'], fontsize=5, rotation=30)
    ax.set_ylabel('Mean(|SHAP value|)', fontsize=AXIS_FONT)
    ax.legend(fontsize=5, frameon=False)
    ax.tick_params(direction='out', pad=1, length=1)

def plot_shap_scatter(ax, X, shap_vals, feature, color_var, xlabel, ylabel):
    """Scatter plot of SHAP values vs feature, color-coded by another variable."""
    # store the scatter artist
    sc = ax.scatter(
        X[feature],
        shap_vals[feature],
        c=X[color_var],
        cmap='coolwarm',
        s=np.sqrt(X[color_var] + 1),
        vmin=-2,
        vmax=2
    )


    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Z-scored detrended BA', fontsize=AXIS_FONT)
    cbar.ax.tick_params(labelsize=5, length=1)


    # Labels, ticks
    ax.set_xlabel(xlabel, fontsize=AXIS_FONT)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONT)
    ax.tick_params(direction='out', pad=1, length=1, labelsize=5)

# ---------------- MAIN ---------------- #
def main():
    fig = plt.figure(figsize=FIGSIZE)
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Compute SHAP for three periods
    shap_df_pre, _, _ = compute_shap_values(PATHS["pre_model"], PATHS["pre_test"])
    shap_df_during, _, _ = compute_shap_values(PATHS["during_model"], PATHS["during_test"])
    shap_df_post, X_post, _ = compute_shap_values(PATHS["post_model"], PATHS["post_test"])

    # Panel a — Bar plot
    ax_a = fig.add_subplot(gs[0, 0])
    plot_shap_bar(ax_a, shap_df_pre, shap_df_during, shap_df_post)
    ax_a.add_artist(AnchoredText("a", loc='upper left', frameon=False, prop=dict(fontsize=PANEL_FONT)))

    # Read SHAP plotting values for post-fire model
    shap_vals_post = pd.read_csv(PATHS["post_plot_vals"])

    # Panel b — T2 scatter
    ax_b = fig.add_subplot(gs[0, 1])
    plot_shap_scatter(ax_b, X_post, shap_vals_post, 'T2', 'BA',
                      "Z-scored detrended post-fire Ta", "SHAP value for Post-fire Ta")
    ax_b.add_artist(AnchoredText("b", loc='upper left', frameon=False, prop=dict(fontsize=PANEL_FONT)))

    # Panel c — Albedo scatter
    ax_c = fig.add_subplot(gs[0, 2])
    plot_shap_scatter(ax_c, X_post, shap_vals_post, 'Albedo', 'BA',
                      "Z-scored detrended post-fire Albedo", "SHAP value for Post-fire Albedo")
    ax_c.add_artist(AnchoredText("c", loc='upper right', frameon=False, prop=dict(fontsize=PANEL_FONT)))

    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
