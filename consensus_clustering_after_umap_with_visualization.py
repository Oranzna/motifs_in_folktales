from __future__ import annotations

import os
from argparse import ArgumentParser

import geodatasets
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import ListedColormap
from shapely.geometry import box
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def read_excel_matrix(
    excel_path: str,
    sheet_name: str | int = 0,
) -> tuple[np.ndarray, pd.Index, list[str]]:
    """
    Reads a binary motif matrix from Excel.

    Expected structure:
      Column 0: story/title IDs
      Columns 1 onward: binary motif columns with values 0 or 1
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    if df.shape[1] < 2:
        raise ValueError(
            "Excel must have at least 2 columns: first column names + >=1 attribute column."
        )

    story_ids = df.iloc[:, 0].astype(str).str.strip()
    X_df = df.iloc[:, 1:]
    motif_names = X_df.columns.astype(str).tolist()

    X_df = X_df.apply(pd.to_numeric, errors="raise")

    vals = pd.unique(X_df.to_numpy().ravel())
    vals = vals[~pd.isna(vals)]
    bad = [v for v in vals if v not in (0, 1, 0.0, 1.0)]

    if bad:
        raise ValueError(
            f"Attributes must be 0/1 only. Found invalid values, sample: {bad[:10]}"
        )

    X = X_df.to_numpy(dtype=float)

    return X, pd.Index(story_ids, name="story_id"), motif_names


def consensus_clustering(
    X: np.ndarray,
    k: int,
    n_runs: int,
    sample_frac: float,
    replace: bool,
    linkage_final: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs consensus clustering.

    Each run samples a subset of rows, applies KMeans, and records how often
    pairs of samples are assigned to the same cluster. A final agglomerative
    clustering is then applied to the consensus distance matrix.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

    n = X.shape[0]

    if not (2 <= k <= n):
        raise ValueError(f"k must be between 2 and n_samples ({n}). Got {k}.")

    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in the interval (0, 1].")

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1.")

    if linkage_final == "ward":
        raise ValueError(
            "linkage_final='ward' cannot be used with metric='precomputed'. "
            "Use 'average', 'complete', or 'single'."
        )

    m = max(2, int(np.floor(sample_frac * n)))
    rng = np.random.default_rng(random_state)

    together = np.zeros((n, n), dtype=float)
    cooccur = np.zeros((n, n), dtype=float)

    for run in range(n_runs):
        idx = rng.choice(n, size=m, replace=replace)
        idx_unique = np.unique(idx)

        if idx_unique.size < k:
            continue

        Xs = X[idx_unique]

        km = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state + run,
        )
        labs = km.fit_predict(Xs)

        cooccur[np.ix_(idx_unique, idx_unique)] += 1.0

        for c in range(k):
            members = idx_unique[labs == c]

            if members.size <= 1:
                continue

            together[np.ix_(members, members)] += 1.0

    C = np.divide(
        together,
        cooccur,
        out=np.zeros_like(together),
        where=cooccur > 0,
    )

    np.fill_diagonal(C, 1.0)

    D_final = 1.0 - C

    final = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage=linkage_final,
    )

    labels_final = final.fit_predict(D_final)

    return labels_final, C


def save_outputs_and_show_heatmap(
    story_ids: pd.Index,
    labels: np.ndarray,
    C: np.ndarray,
    out_prefix: str,
    out_dir: str,
) -> None:
    """
    Saves cluster labels and consensus matrix to Excel.
    Also displays a heatmap of the consensus matrix sorted by final cluster labels.
    """
    os.makedirs(out_dir, exist_ok=True)

    labels_df = pd.DataFrame(
        {
            "story_id": story_ids.astype(str),
            "cluster": labels,
        }
    )

    labels_path = os.path.join(out_dir, f"{out_prefix}_umap_cluster_labels.xlsx")
    labels_df.to_excel(labels_path, index=False)

    C_df = pd.DataFrame(
        C,
        index=story_ids.astype(str),
        columns=story_ids.astype(str),
    )

    C_path = os.path.join(out_dir, f"{out_prefix}_umap_cons_matrix.xlsx")
    C_df.to_excel(C_path)

    order = np.argsort(labels)
    C_sorted = C[np.ix_(order, order)]

    plt.figure(figsize=(7, 6))
    plt.imshow(C_sorted, aspect="auto")
    plt.colorbar(label="Consensus P[same cluster]")
    plt.title("Consensus matrix sorted by final cluster labels")
    plt.xlabel("Stories sorted")
    plt.ylabel("Stories sorted")
    plt.tight_layout()
    plt.show()

    print("Saved:")
    print(" -", os.path.abspath(labels_path))
    print(" -", os.path.abspath(C_path))


def get_fairy_clusters(
    list_of_names: list[str] | pd.Index,
    km_labels: np.ndarray,
    n_clusters: int,
) -> dict[int, list[str]]:
    """
    Returns a dictionary where each cluster ID maps to the story titles in that cluster.
    """
    names = list(map(str, list_of_names))

    if len(names) != len(km_labels):
        raise ValueError(
            f"Number of names ({len(names)}) does not match number of labels ({len(km_labels)})."
        )

    stories_dictionary: dict[int, list[str]] = {}

    for i in range(n_clusters):
        indices = np.where(km_labels == i)[0]
        stories_dictionary[i] = [names[j] for j in indices]

    return stories_dictionary


def save_clusters_to_file(
    fairy_dictionary: dict[int, list[str]],
    file_name: str,
) -> None:
    """
    Saves cluster membership to a text file.
    """
    with open(file_name, "w", encoding="utf-8") as w_file:
        for key in sorted(fairy_dictionary.keys()):
            w_file.write(f"{key}:\n")

            for story in fairy_dictionary[key]:
                w_file.write(f"  - {story}\n")

            w_file.write("\n")


def visualize_clusters(
    cluster_labels: np.ndarray,
    embedding: np.ndarray,
    my_cmap: ListedColormap,
) -> None:
    """
    Shows a 2D scatterplot of the UMAP embedding colored by cluster label.
    """
    cluster_labels = np.asarray(cluster_labels)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap=my_cmap,
        s=150,
        edgecolor="black",
        linewidth=0.5,
    )

    unique_labels = np.unique(cluster_labels)

    colorbar = plt.colorbar(
        scatter,
        boundaries=np.arange(unique_labels.max() + 2) - 0.5,
    )
    colorbar.set_ticks(np.arange(unique_labels.max() + 1))

    plt.title("UMAP plot showing consensus clustering")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.tight_layout()
    plt.show()


def create_file_geolocation(
    file_path: str,
    story_ids: pd.Index,
    cluster_labels: np.ndarray,
    out_xlsx: str,
) -> gpd.GeoDataFrame:
    """
    Reads geodata from Excel and adds the newly computed cluster labels.

    Expected Excel columns:
      Title
      Country
      Latitude
      Longitude
      Continent

    The function matches rows by Title, not by row order.
    Existing Cluster values are replaced with the newly computed cluster labels.
    """
    df = pd.read_excel(file_path)

    required_cols = ["Title", "Country", "Latitude", "Longitude", "Continent"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in geodata file: {missing}")

    if len(story_ids) != len(cluster_labels):
        raise ValueError(
            f"story_ids length ({len(story_ids)}) does not match "
            f"cluster_labels length ({len(cluster_labels)})."
        )

    df["Title"] = df["Title"].astype(str).str.strip()

    labels_df = pd.DataFrame(
        {
            "Title": pd.Index(story_ids).astype(str).str.strip(),
            "Cluster": cluster_labels,
        }
    )

    if labels_df["Title"].duplicated().any():
        duplicates = labels_df.loc[labels_df["Title"].duplicated(), "Title"].tolist()
        raise ValueError(
            f"Duplicate titles found in clustering input. Examples: {duplicates[:10]}"
        )

    if df["Title"].duplicated().any():
        duplicates = df.loc[df["Title"].duplicated(), "Title"].tolist()
        raise ValueError(
            f"Duplicate titles found in geodata file. Examples: {duplicates[:10]}"
        )

    merged = df.merge(
        labels_df,
        on="Title",
        how="left",
        suffixes=("", "_new"),
    )

    if "Cluster_new" in merged.columns:
        merged["Cluster"] = merged["Cluster_new"]
        merged.drop(columns=["Cluster_new"], inplace=True)

    missing_clusters = merged["Cluster"].isna().sum()

    if missing_clusters > 0:
        missing_titles = merged.loc[merged["Cluster"].isna(), "Title"].tolist()
        raise ValueError(
            f"{missing_clusters} rows in geodata file did not match any story_id. "
            f"Examples: {missing_titles[:10]}"
        )

    merged["Latitude"] = pd.to_numeric(merged["Latitude"], errors="raise")
    merged["Longitude"] = pd.to_numeric(merged["Longitude"], errors="raise")
    merged["Cluster"] = merged["Cluster"].astype(int)

    geo_data = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(
            merged["Longitude"],
            merged["Latitude"],
        ),
    )

    geo_data.to_excel(out_xlsx, index=False)

    print(" -", os.path.abspath(out_xlsx))

    return geo_data


def create_world_map(
    geo_data: gpd.GeoDataFrame,
    n_clusters: int,
    my_cmap: ListedColormap,
) -> None:
    """
    Shows a world map with story locations colored by cluster.
    """
    path = geodatasets.get_path("naturalearth.land")
    world_map = gpd.read_file(path)

    fig, ax = plt.subplots(figsize=(14, 10))
    world_map.plot(ax=ax, color="antiquewhite", edgecolor="antiquewhite")

    for cluster_group in range(n_clusters):
        cluster_data = geo_data[geo_data["Cluster"] == cluster_group]

        if cluster_data.empty:
            continue

        cluster_color = my_cmap(
            cluster_group / (n_clusters - 1 if n_clusters > 1 else 1)
        )

        cluster_data.plot(
            ax=ax,
            marker="o",
            facecolor=cluster_color,
            edgecolor="black",
            linewidth=0.5,
            markersize=150,
            label=f"Cluster {cluster_group}",
        )

    plt.title(f"Cinderella types around the world - {n_clusters} clusters")
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_europe_map(
    geo_data: gpd.GeoDataFrame,
    n_clusters: int,
    my_cmap: ListedColormap,
) -> None:
    """
    Shows a Europe-only map with story locations colored by cluster.
    """
    path = geodatasets.get_path("naturalearth.land")
    world_map = gpd.read_file(path)

    europe_box = box(-25.0, 35.0, 45.0, 70.0)
    europe_box_gdf = gpd.GeoDataFrame({"geometry": [europe_box]}, crs=world_map.crs)
    europe_land = gpd.clip(world_map, europe_box_gdf)

    fig, ax = plt.subplots(figsize=(14, 10))
    europe_land.plot(ax=ax, color="antiquewhite", edgecolor="antiquewhite")

    europe = geo_data[geo_data["Continent"].astype(str).str.strip() == "Europe"].copy()

    for cluster_group in range(n_clusters):
        eu_cluster_data = europe[europe["Cluster"] == cluster_group]

        if eu_cluster_data.empty:
            continue

        eu_cluster_color = my_cmap(
            cluster_group / (n_clusters - 1 if n_clusters > 1 else 1)
        )

        eu_cluster_data.plot(
            ax=ax,
            marker="o",
            facecolor=eu_cluster_color,
            edgecolor="black",
            linewidth=0.5,
            markersize=150,
            label=f"Cluster {cluster_group}",
        )

    plt.title(f"Cinderella types in Europe - {n_clusters} clusters")
    plt.legend()
    plt.tight_layout()
    plt.show()


def pac(
    C: np.ndarray,
    u1: float = 0.1,
    u2: float = 0.9,
) -> float:
    """
    Computes the Proportion of Ambiguous Clustering.

    Lower PAC means the consensus matrix has more values near 0 or 1,
    which usually indicates more stable clustering.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix.")

    n = C.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = C[mask]

    return float(np.mean((vals > u1) & (vals < u2)))


def print_motifs_per_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    motif_names: list[str],
    top_n: int | None = None,
    min_count: int = 0,
    sort_by: str = "count",
) -> None:
    """
    Prints motif counts and percentages per cluster.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) != labels length ({labels.shape[0]}).")

    if X.shape[1] != len(motif_names):
        raise ValueError(
            f"X cols ({X.shape[1]}) != len(motif_names) ({len(motif_names)})."
        )

    for cl in sorted(np.unique(labels)):
        idx = np.where(labels == cl)[0]
        n_items = idx.size

        if n_items == 0:
            continue

        counts = X[idx].sum(axis=0).astype(int)
        perc = counts / n_items

        rows = list(zip(motif_names, counts.tolist(), perc.tolist()))

        if min_count > 0:
            rows = [r for r in rows if r[1] >= min_count]

        if sort_by == "count":
            rows.sort(key=lambda t: t[1], reverse=True)
        elif sort_by == "percentage":
            rows.sort(key=lambda t: t[2], reverse=True)
        elif sort_by == "name":
            rows.sort(key=lambda t: t[0])
        else:
            raise ValueError("sort_by must be one of: 'count', 'percentage', 'name'.")

        if top_n is not None:
            rows = rows[:top_n]

        print(f"\nCluster {cl}:")
        print(f"number of items: {n_items}")

        for name, c, p in rows:
            print(f"{name}: {c}  Percentage: {p:.2f}")


def parse_arguments():
    parser = ArgumentParser(
        description="Run UMAP + consensus clustering on a binary fairy-tale motif matrix."
    )

    parser.add_argument(
        "--excel_path",
        required=True,
        help="Path to the Excel file containing the binary motif matrix.",
    )

    parser.add_argument(
        "--geo_excel_path",
        required=True,
        help="Path to the Excel file containing geodata.",
    )

    parser.add_argument(
        "--out_dir",
        default="results",
        help="Folder where output files will be saved. Default: results",
    )

    parser.add_argument(
        "--out_prefix",
        default=None,
        help="Prefix for output files. If omitted, one is created automatically.",
    )

    parser.add_argument(
        "--sheet_name",
        default=0,
        help="Excel sheet name or index. Default: 0",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # -----------------------------
    # Settings you can edit
    # -----------------------------
    K = 4
    N_RUNS = 1000
    SAMPLE_FRAC = 0.9
    REPLACE = True
    LINKAGE_FINAL = "average"
    RANDOM_STATE = 42

    N_NEIGHBORS = 6
    MIN_DIST = 0.1

    custom_colors = ["#8dd3c7", "#ffd92f", "#bebada", "#fb8072"]
    my_cmap = ListedColormap(custom_colors)

    # -----------------------------
    # Paths from command line
    # -----------------------------
    excel_path = args.excel_path
    geo_excel_path = args.geo_excel_path
    sheet_name = args.sheet_name
    out_dir = args.out_dir
    out_prefix = args.out_prefix or f"consensus_umap_{K}_clusters"

    if len(custom_colors) < K:
        raise ValueError(
            f"You provided {len(custom_colors)} colors, but K={K}. "
            "Add more colors to custom_colors."
        )

    X, story_ids, motif_names = read_excel_matrix(
        excel_path,
        sheet_name=sheet_name,
    )

    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )

    Z = reducer.fit_transform(X)

    labels, C = consensus_clustering(
        X=Z,
        k=K,
        n_runs=N_RUNS,
        sample_frac=SAMPLE_FRAC,
        replace=REPLACE,
        linkage_final=LINKAGE_FINAL,
        random_state=RANDOM_STATE,
    )

    sil_umap = silhouette_score(
        Z,
        labels,
        metric="euclidean",
    )
    print(f"Silhouette UMAP, Euclidean: {sil_umap:.4f}")

    D_final = 1.0 - C

    sil_c = silhouette_score(
        D_final,
        labels,
        metric="precomputed",
    )
    print(f"Silhouette 1-C, precomputed: {sil_c:.4f}")

    pac_value = pac(C)
    print(f"Proportion of ambiguous clustering: {pac_value:.4f}")

    counts = pd.Series(labels).value_counts().sort_index()

    print("\nStories per cluster:")
    for cl, n_ in counts.items():
        print(f"  Cluster {cl}: {n_}")

    print(f"  Total: {counts.sum()}\n")

    print_motifs_per_cluster(
        X=X,
        labels=labels,
        motif_names=motif_names,
        top_n=None,
        min_count=1,
        sort_by="count",
    )

    save_outputs_and_show_heatmap(
        story_ids=story_ids,
        labels=labels,
        C=C,
        out_prefix=out_prefix,
        out_dir=out_dir,
    )

    clusters_dict = get_fairy_clusters(
        list_of_names=story_ids,
        km_labels=labels,
        n_clusters=K,
    )

    clusters_txt_path = os.path.join(
        out_dir,
        f"{out_prefix}_clusters.txt",
    )

    save_clusters_to_file(
        fairy_dictionary=clusters_dict,
        file_name=clusters_txt_path,
    )

    print(" -", os.path.abspath(clusters_txt_path))

    visualize_clusters(
        cluster_labels=labels,
        embedding=Z,
        my_cmap=my_cmap,
    )

    geo_out_xlsx = os.path.join(
        out_dir,
        f"umap_{out_prefix}_geodata.xlsx",
    )

    geo_data = create_file_geolocation(
        file_path=geo_excel_path,
        story_ids=story_ids,
        cluster_labels=labels,
        out_xlsx=geo_out_xlsx,
    )

    create_world_map(
        geo_data=geo_data,
        n_clusters=K,
        my_cmap=my_cmap,
    )

    create_europe_map(
        geo_data=geo_data,
        n_clusters=K,
        my_cmap=my_cmap,
    )