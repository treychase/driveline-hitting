# Plot histogram

def plot_histogram(data, bins=30):
    import matplotlib.pyplot as plt
    import pandas as pd
 
    if isinstance(data, pd.Series):
        data = data.to_frame()
 
    cols = list(data.columns)
    ncols = min(3, len(cols))
    nrows = -(-len(cols) // ncols)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = [axes] if len(cols) == 1 else axes.flatten()
 
    for ax, col in zip(axes, cols):
        values = data[col].dropna()
        mean = values.mean()
        sd = values.std()
        label = col.replace("_", " ").title()
 
        counts, bin_edges, patches = ax.hist(values, bins=bins, color="lightblue", edgecolor="black")
 
        lower, upper = mean - 2 * sd, mean + 2 * sd
        for patch, left_edge, right_edge in zip(patches, bin_edges[:-1], bin_edges[1:]):
            bin_center = (left_edge + right_edge) / 2
            if bin_center < lower or bin_center > upper:
                patch.set_facecolor("orange")
 
        ax.axvline(mean, color="red", linestyle="--")
        ax.text(0.97, 0.97, f"mean = {mean:.2f}\nsd = {sd:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
 
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {label}")
        ax.grid(False)
 
    for ax in axes[len(cols):]:
        ax.axis("off")
 
    plt.tight_layout()
    plt.show()

# Correlation Plot
def plot_correlation(data):
    import matplotlib.pyplot as plt
 
    corr = data.corr(numeric_only=True)
    labels = [c.replace("_", " ").title() for c in corr.columns]
 
    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 3, 0.6 * len(labels) + 3))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
 
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
 
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
 
    ax.set_title("Correlation Between Variables")
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.grid(False)
    plt.tight_layout()
    plt.show()
 
 


def plot_keyword_summary(summary_df):
    import matplotlib.pyplot as plt
    

    sorted_df = summary_df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(sorted_df) * 0.4)))
    bars = ax.barh(sorted_df["keyword"], sorted_df["count"], color="#2b6ca3")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va="center", fontsize=9)

    ax.set_xlabel("Column Count")
    ax.set_title("Column Counts by Keyword")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()