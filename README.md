# Folktale Motif Analysis

This project analyzes folktales based on motif presence. It first checks whether selected motifs are present in each tale, converts the answers into binary motif data, and then uses the motif matrix for clustering and comparison.

## Workflow

The project contains several scripts:

1. `check_motifs.py`  
   Checks whether selected motifs are present in each tale. The script uses `.txt` tale files and the OpenAI API.

2. `prompts.py`  
   Contains the prompts used by `check_motifs.py`.

3. `change_to_binary_data.py`  
   Converts motif answers into a binary matrix, where `1` means the motif is present and `0` means it is absent.

4. `consensus_clustering_after_umap_with_visualization.py`  
   Clusters the original tales based on the binary motif matrix. The script first reduces dimensionality with UMAP, then applies consensus clustering and visualizes the results. The motif Excel files should have this structure:

```text
Title | Motif 1 | Motif 2 | Motif 3 | ...
```

The script uses a geodata Excel file to visualize tale locations on a map. This file links each tale title to its geographic information.

The geodata file should contain at least these columns:

```text
Title | Country | Latitude | Longitude | Continent
Tale1 | Germany | 51.1657 | 10.4515 | Europe
Tale2 | France | 46.2276 | 2.2137 | Europe
```

5. `assign_new_stories_to_clusters.py`  
   Projects new tales into the existing UMAP space and assigns them to the closest existing cluster.

