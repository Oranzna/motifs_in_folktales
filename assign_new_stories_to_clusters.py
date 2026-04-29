import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import umap
import geopandas as gpd


# =========================
# SETTINGS
# =========================

n_clusters = 4   # set the nubmer of clusters

# Optional: restrict original dataset to first N rows
n_count = None   # set to None to use all rows

# Column names
title_col = "Title"
latitude_col = "Latitude"
longitude_col = "Longitude"
continent_col = "Continent"

# Custom colours
custom_colors = ["#8dd3c7", "#ffd92f", "#bebada", "#fb8072"]
my_cmap = ListedColormap(custom_colors)


# =========================
# ARGUMENTS
# =========================

def parse_arguments():
    parser = ArgumentParser(
        description="Cluster original tales and assign new stories to the existing clusters."
    )

    parser.add_argument(
        "--foreign_path",
        required=True,
        help="Path to the Excel file with the original motif matrix.",
    )

    parser.add_argument(
        "--new_stories_path",
        required=True,
        help="Path to the Excel file with the new stories to assign to clusters.",
    )

    parser.add_argument(
        "--geo_path",
        required=True,
        help="Path to the Excel file with foreign story geodata.",
    )

    parser.add_argument(
        "--out_dir",
        default="results",
        help="Folder where output files will be saved.",
    )

    return parser.parse_args()


# =========================
# HELPERS
# =========================

def load_dataframe(file_path, n_rows=None):
    df = pd.read_excel(file_path)
    if n_rows is not None:
        df = df.iloc[:n_rows].copy()
    return df


def infer_motif_columns(df, title_column="Title"):
    motif_cols = [col for col in df.columns if col != title_column]
    if not motif_cols:
        raise ValueError("No motif columns found.")
    return motif_cols


def validate_columns(df, required_cols, df_name="DataFrame"):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def return_n_matrix(df, motif_cols, title_column):
    validate_columns(df, [title_column] + motif_cols, "Original data")
    fairy_array = df[motif_cols].to_numpy()
    fairy_titles = df[title_column].astype(str).tolist()
    return fairy_array, fairy_titles


def load_additional_stories(file_path, motif_cols, title_column):
    df = pd.read_excel(file_path)
    validate_columns(df, [title_column], "Additional stories data")

    missing = [col for col in motif_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Additional stories file is missing motif columns: {missing}")

    fairy_array = df[motif_cols].to_numpy()
    return fairy_array, df


def use_umap(fairy_array):
    reducer = umap.UMAP(
        n_neighbors=6,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(fairy_array)
    return reducer, embedding


def perform_clustering(embedding, original_matrix, motif_names):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_cluster_labels = kmeans.fit_predict(embedding)

    print("\n=== Motif counts per cluster ===")
    for cluster_value in range(n_clusters):
        cluster_rows = original_matrix[km_cluster_labels == cluster_value]

        print(f"\nCluster {cluster_value}:")
        print("number of items:", len(cluster_rows))

        if len(cluster_rows) == 0:
            print("Empty cluster.")
            continue

        motif_numbers = np.sum(cluster_rows == 1, axis=0)

        for i, count_n in enumerate(motif_numbers):
            percentage = count_n / len(cluster_rows)
            print(f"{motif_names[i]}: {count_n} Percentage: {percentage:.2f}")

    return kmeans, km_cluster_labels


def visualize_clusters(cluster_labels, embedding):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap=my_cmap,
        s=150,
        edgecolor="black",
        linewidth=0.5
    )
    cbar = plt.colorbar(
        scatter,
        boundaries=np.arange(np.max(cluster_labels) + 2) - 0.5
    )
    cbar.set_ticks(np.arange(np.max(cluster_labels) + 1))
    plt.title("UMAP projection of KMeans clustering results")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.show()


def calculate_silhouette_score(embedding, cluster_labels):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(embedding) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(embedding, cluster_labels)
    print(f"\nFor n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.4f}")

    sample_silhouette_values = silhouette_samples(embedding, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"The silhouette plot for {n_clusters} clusters is {silhouette_avg:.4f}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle("Silhouette analysis for KMeans clustering", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def get_fairy_clusters(list_of_titles, km_labels):
    stories_dictionary = {}
    for i in range(n_clusters):
        indices = np.where(km_labels == i)[0]
        stories_in_cluster = [list_of_titles[j] for j in indices]
        stories_dictionary[i] = stories_in_cluster
    return stories_dictionary


def save_clusters_to_file(fairy_dictionary, file_name):
    with open(file_name, "w", encoding="utf-8") as w_file:
        for key, values in fairy_dictionary.items():
            w_file.write(f"Cluster {key}:\n")
            for value in values:
                w_file.write(f"  - {value}\n")
            w_file.write("\n")


def create_file_geolocation(geo_file_path, titles, cluster_labels, output_file):
    geo_df = pd.read_excel(geo_file_path)
    validate_columns(
        geo_df,
        [title_col, continent_col, latitude_col, longitude_col],
        "Geolocation data"
    )

    title_to_cluster = pd.DataFrame({
        title_col: titles,
        "Cluster": cluster_labels
    })

    merged = geo_df.merge(title_to_cluster, on=title_col, how="inner")

    if len(merged) != len(title_to_cluster):
        print(
            f"Warning: only {len(merged)} of {len(title_to_cluster)} foreign stories "
            f"matched rows in the geolocation file."
        )

    geo_data = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged[longitude_col], merged[latitude_col])
    )

    geo_data.to_excel(output_file, index=False)
    return geo_data


def jitter_points(arr, scale=0.03, random_state=42):
    # Many stories may overlap on the map/plot, so we add jitter to show there are more stories there.
    rng = np.random.default_rng(random_state)
    return arr + rng.normal(0, scale, arr.shape)


def visualize_all_stories(umap_old, labels_old, umap_new, labels_new, cmap=my_cmap):
    plt.figure(figsize=(10, 8))

    vmin, vmax = 0, n_clusters - 1

    # Jitter new stories only, for display
    umap_new_plot = jitter_points(umap_new, scale=0.04, random_state=123)

    plt.scatter(
        umap_old[:, 0], umap_old[:, 1],
        c=labels_old, cmap=cmap, vmin=vmin, vmax=vmax,
        s=100, edgecolor="black", linewidth=0.5, alpha=0.65
    )

    plt.scatter(
        umap_new_plot[:, 0], umap_new_plot[:, 1],
        c=labels_new, cmap=cmap, vmin=vmin, vmax=vmax,
        s=120, edgecolor="black", linewidth=1.2, marker="s", alpha=0.9
    )

    cbar = plt.colorbar(boundaries=np.arange(vmax + 2) - 0.5)
    cbar.set_ticks(np.arange(vmax + 1))

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=1.2,
            markersize=np.sqrt(100),
            label="Foreign"
        ),
        Line2D(
            [0], [0],
            marker="s",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=1.2,
            markersize=np.sqrt(120),
            label="New stories"
        )
    ]

    plt.legend(handles=legend_elements, loc="best")
    plt.title("UMAP projection of KMeans clustering results")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)

    f_path = args.foreign_path
    new_stories_path = args.new_stories_path
    geo_path = args.geo_path

    geo_output_path = os.path.join(
        args.out_dir,
        f"geodata_{n_clusters}_clusters.xlsx"
    )

    clusters_output_path = os.path.join(
        args.out_dir,
        f"foreign_tales_{n_clusters}_clusters.txt"
    )

    new_stories_output_path = os.path.join(
        args.out_dir,
        f"added_new_tales_{n_clusters}_clusters.xlsx"
    )

    # Load foreign stories
    original_df = load_dataframe(f_path, n_rows=n_count)

    # Infer motif columns from foreign file
    motif_columns = infer_motif_columns(original_df, title_column=title_col)

    # Build original matrix
    n_matrix, stories_list = return_n_matrix(original_df, motif_columns, title_col)

    # Fit UMAP on foreign stories
    reducer, umap_matrix = use_umap(n_matrix)

    # Fit KMeans in UMAP space
    kmeans, km_cluster_labels = perform_clustering(umap_matrix, n_matrix, motif_columns)

    # Evaluate + visualize foreign clustering
    calculate_silhouette_score(umap_matrix, km_cluster_labels)
    visualize_clusters(km_cluster_labels, umap_matrix)

    # Save foreign cluster memberships
    similar_stories = get_fairy_clusters(stories_list, km_cluster_labels)
    save_clusters_to_file(similar_stories, clusters_output_path)

    # Create geodata only for foreign stories
    map_data = create_file_geolocation(
        geo_path,
        stories_list,
        km_cluster_labels,
        geo_output_path
    )

    # Load new stories using same motif columns by name
    new_stories_array, new_df = load_additional_stories(
        new_stories_path,
        motif_columns,
        title_col
    )

    # Project new stories into foreign UMAP space
    new_umap_matrix = reducer.transform(new_stories_array)

    # Assign new stories to existing clusters
    new_cluster_labels = kmeans.predict(new_umap_matrix)

    print("\n=== New story assignments ===")
    unique_clusters, counts = np.unique(new_cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster_id}: {count} new stories")

    print("\n=== Plotting checks for new stories ===")
    print("Total new stories:", len(new_df))
    print("Total assigned labels:", len(new_cluster_labels))
    print("Total rows sent to scatter:", len(new_umap_matrix))

    unique_umap = np.unique(new_umap_matrix.round(6), axis=0)
    print("Distinct plotted UMAP positions:", len(unique_umap))

    # UMAP plot with foreign + new stories
    visualize_all_stories(
        umap_matrix,
        km_cluster_labels,
        new_umap_matrix,
        new_cluster_labels
    )

    # Save new story assignments
    new_df["Assigned Cluster"] = new_cluster_labels
    new_df.to_excel(new_stories_output_path, index=False)

    print("\nSaved:")
    print(" -", os.path.abspath(clusters_output_path))
    print(" -", os.path.abspath(geo_output_path))
    print(" -", os.path.abspath(new_stories_output_path))

    print("\nDone.")