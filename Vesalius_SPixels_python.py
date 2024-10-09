import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import scipy
from shapely.geometry import Polygon
from shapely.vectorized import contains  # Ensure Shapely version >= 2.0
from collections import Counter
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_embeddings(
    adata,
    dim_reduction='PCA',
    normalization='log_norm',
    use_counts='raw',
    dimensions=30,
    tensor_resolution=1,
    filter_grid=0.01,
    filter_threshold=0.995,
    nfeatures=2000,
    features=None,
    min_cutoff='q5',
    remove_lsi_1=True,
    n_jobs=None,
    verbose=True
):
    """
    Generate embeddings for the given AnnData object using specified parameters.

    Parameters:
    - adata: AnnData object containing the data.
    - dim_reduction: Method for dimensionality reduction ('PCA', 'UMAP', etc.).
    - normalization: Normalization method ('log_norm', 'SCT', 'TFIDF', 'none').
    - use_counts: Which counts to use ('raw' or layer name).
    - dimensions: Number of dimensions for reduction.
    - tensor_resolution: Resolution parameter for tensor.
    - filter_grid: Grid filtering parameter.
    - filter_threshold: Threshold for filtering tiles.
    - nfeatures: Number of features to select.
    - features: Specific features to use.
    - min_cutoff: Minimum cutoff for feature selection.
    - remove_lsi_1: Whether to remove the first LSI component.
    - n_jobs: Number of parallel jobs.
    - verbose: Whether to print verbose messages.

    Returns:
    - adata: AnnData object with embeddings added.
    """
    if verbose:
        print("Starting generate_embeddings...")

    # Generate tiles if not already generated
    if 'tiles_generated' not in adata.uns or not adata.uns['tiles_generated']:
        if verbose:
            print("Tiles not found. Generating tiles...")
        adata = generate_tiles(
            adata,
            tensor_resolution=tensor_resolution,
            filter_grid=filter_grid,
            filter_threshold=filter_threshold,
            verbose=verbose,
            n_jobs=n_jobs
        )

    # Process counts with specified normalization
    if verbose:
        print("Processing counts...")
    adata_proc = process_counts(
        adata,
        method=normalization,
        use_counts=use_counts,
        nfeatures=nfeatures,
        min_cutoff=min_cutoff,
        verbose=verbose
    )

    adata.layers['log_norm'] = adata_proc.X.copy()
    # Embed latent space using specified dimensionality reduction
    if verbose:
        print("Embedding latent space...")
    embeds = embed_latent_space(
        adata_proc,
        dim_reduction=dim_reduction,
        dimensions=dimensions,
        features=features,
        remove_lsi_1=remove_lsi_1,
        verbose=verbose,
        n_jobs=n_jobs
    )

    # Store embeddings in AnnData object
    adata.obsm['X_embedding'] = embeds
    adata.uns['embedding_method'] = dim_reduction
    if verbose:
        print("generate_embeddings completed.")

    return adata

def generate_tiles(
    adata,
    tensor_resolution=1,
    filter_grid=0.01,
    filter_threshold=0.995,
    verbose=True,
    n_jobs=None
):
    """
    Generate spatial tiles for the given AnnData object.

    Parameters:
    - adata: AnnData object containing spatial data.
    - tensor_resolution: Resolution parameter for tensor.
    - filter_grid: Grid filtering parameter.
    - filter_threshold: Threshold for filtering tiles.
    - verbose: Whether to print verbose messages.
    - n_jobs: Number of parallel jobs.

    Returns:
    - adata: AnnData object with tiles generated.
    """
    # Check if tiles are already generated
    if 'tiles_generated' in adata.uns and adata.uns['tiles_generated']:
        if verbose:
            print("Tiles have already been generated. Skipping tile generation.")
        return adata

    if verbose:
        print("Starting generate_tiles...")

    # Copy original spatial coordinates
    coordinates = pd.DataFrame(adata.obsm['spatial'])
    coordinates.index = adata.obs.index.copy()
    if verbose:
        print(f"Original coordinates: {coordinates.shape}")

    # Apply grid filtering if specified
    if 0 < filter_grid < 1:
        if verbose:
            print("Filtering outlier beads...")
        coordinates = filter_grid_function(coordinates, filter_grid)
        if verbose:
            print(f"Coordinates after filtering: {coordinates.shape}")

    # Reduce tensor resolution if specified
    if 0 < tensor_resolution < 1:
        if verbose:
            print("Reducing tensor resolution...")
        coordinates = reduce_tensor_resolution(coordinates, tensor_resolution)
        if verbose:
            print(f"Coordinates after resolution reduction: {coordinates.shape}")

    # Perform Voronoi tessellation
    if verbose:
        print("Performing Voronoi tessellation...")
    from scipy.spatial import Voronoi
    vor = Voronoi(coordinates)
    if verbose:
        print("Voronoi tessellation completed.")

    # Filter tiles based on area
    if verbose:
        print("Filtering tiles...")
    filtered_regions, filtered_coordinates, index = filter_tiles(vor, coordinates, filter_threshold)
    if verbose:
        print(f"Filtered regions: {len(filtered_regions)}, Filtered coordinates: {filtered_coordinates.shape}")

    # Rasterize tiles with parallel processing
    if verbose:
        print("Rasterising tiles...")
    tiles = rasterise(filtered_regions, filtered_coordinates, index, vor, n_jobs=n_jobs)
    if verbose:
        print(f"Rasterisation completed. Number of tiles: {len(tiles)}")

    # Store tiles in AnnData object
    adata.uns['tiles'] = tiles
    adata.uns['tiles_generated'] = True
    if verbose:
        print("Tiles have been stored in adata.uns['tiles'].")
    filtered_barcodes = tiles['barcode'].unique()
    initial_obs = adata.n_obs
    adata = adata[filtered_barcodes, :].copy()
    final_obs = adata.n_obs

    if verbose:
        print(f"adata has been subset from {initial_obs} to {final_obs} observations based on filtered tiles.")

    if verbose:
        print("generate_tiles completed.")

    return adata

def filter_grid_function(coordinates, filter_grid):
    """
    Filter out grid coordinates based on the specified grid filter.

    Parameters:
    - coordinates: Array of spatial coordinates.
    - filter_grid: Grid filtering parameter.

    Returns:
    - filtered_coordinates: Filtered array of spatial coordinates.
    """
    grid_x = np.round(coordinates[0] * filter_grid)
    grid_y = np.round(coordinates[1] * filter_grid)
    
    # Use numpy.char.add for string concatenation
    grid_coord = np.char.add(grid_x.astype(str), "_")
    grid_coord = np.char.add(grid_coord, grid_y.astype(str))

    # Count occurrences in each grid cell
    grid_counts = Counter(grid_coord)
    counts = np.array(list(grid_counts.values()))
    threshold = np.quantile(counts, 0.01)

    # Identify low-count grid cells
    low_count_grids = {k for k, v in grid_counts.items() if v <= threshold}

    # Create mask to filter out low-count grid cells
    mask = np.array([gc not in low_count_grids for gc in grid_coord])
    filtered_coordinates = coordinates[mask]

    return filtered_coordinates

def reduce_tensor_resolution(coordinates, tensor_resolution=1):
    """
    Reduce tensor resolution of spatial coordinates.

    Parameters:
    - coordinates: Array of spatial coordinates.
    - tensor_resolution: Resolution parameter for tensor.

    Returns:
    - filtered_coordinates: Reduced resolution coordinates.
    """
    coordinates[0] = np.round(coordinates[0] * tensor_resolution) + 1
    coordinates[1] = np.round(coordinates[1] * tensor_resolution) + 1

    # Group by reduced coordinates and compute mean
    coords_df = coordinates.rename(columns={0: 'x', 1: 'y'})
    coords_df['coords'] = coords_df['x'].astype(str) + '_' + coords_df['y'].astype(str)
    coords_df['original_index'] = coords_df.index
    
    coords_grouped = coords_df.groupby('coords').mean(numeric_only=True).reset_index()
    coords_grouped['original_index'] = coords_df.groupby('coords')['original_index'].first().values
    coords_grouped.set_index('original_index', inplace=True)
    filtered_coordinates = coords_grouped[['x', 'y']]

    return filtered_coordinates

def filter_tiles(vor, coordinates, filter_threshold):
    """
    Filter Voronoi regions based on area threshold.

    Parameters:
    - vor: Voronoi object.
    - coordinates: Array of spatial coordinates.
    - filter_threshold: Threshold for filtering tiles based on area.

    Returns:
    - filtered_regions: List of filtered Voronoi region indices.
    - filtered_coordinates: Array of filtered spatial coordinates.
    """
    point_region = vor.point_region
    regions = vor.regions
    vertices = vor.vertices

    areas = []
    valid_points = []
    valid_regions = []
    index = []
    # Calculate areas of Voronoi regions
    for idx, region_index in tqdm(enumerate(point_region), total=len(point_region), desc="Calculating Voronoi region areas"):
        region = regions[region_index]
        if -1 in region or len(region) == 0:
            continue  # Skip regions with infinite vertices or empty regions
        polygon = Polygon(vertices[region])
        area = polygon.area
        areas.append(area)
        valid_points.append(coordinates.iloc[idx])
        valid_regions.append(region)
        index.append(coordinates.index[idx])
    areas = np.array(areas)
    max_area = np.quantile(areas, filter_threshold)

    filtered_regions = []
    filtered_points = []
    filtered_index = []
    # Filter regions based on area
    for area, region, point, index in tqdm(zip(areas, valid_regions, valid_points, index), total=len(areas), desc="Filtering tiles by area"):
        if area <= max_area:
            filtered_regions.append(region)
            filtered_points.append(point)
            filtered_index.append(index)

    return filtered_regions, np.array(filtered_points), filtered_index

def rasterise(filtered_regions, filtered_points, index, vor, n_jobs=None):
    """
    Rasterize Voronoi regions into grid tiles.

    Parameters:
    - filtered_regions: List of filtered Voronoi region indices.
    - filtered_points: Array of filtered spatial coordinates.
    - vor: Voronoi object.
    - n_jobs: Number of parallel jobs.

    Returns:
    - tiles: DataFrame containing rasterized tiles.
    """
    # Set default number of jobs if not specified
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 0:
        n_jobs = n_jobs
    elif n_jobs == 0:
        n_jobs = 1

    # Create iterable of arguments for rasterization
    iterable = zip(filtered_regions, filtered_points, index, itertools.repeat(vor))
    desc = "Rasterising tiles"

    # Calculate total number of regions
    total = len(filtered_regions)

    # Process rasterization in parallel or serially using the optimized helper function
    tiles_list = _process_in_parallel_map(
        rasterise_single_tile,
        iterable,
        desc,
        n_jobs,
        chunksize=1000,
        total=total
    )

    # Concatenate all tile DataFrames
    if tiles_list:
        tiles = pd.concat(tiles_list, ignore_index=True)
    else:
        tiles = pd.DataFrame(columns=['x', 'y', 'barcode', 'origin_x', 'origin_y', 'origin'])

    return tiles

def _process_in_parallel_map(function, iterable, desc, n_jobs, chunksize=1000, total=None):
    """
    Helper function to process tasks in parallel with a progress bar using map.
    If n_jobs == 1, processes serially.

    Parameters:
    - function: Function to apply to each item.
    - iterable: Iterable of items to process.
    - desc: Description for the progress bar.
    - n_jobs: Number of parallel jobs.
    - chunksize: Number of tasks to submit in each batch.
    - total: Total number of items in the iterable.

    Returns:
    - results: List of results after applying the function.
    """
    results = []
    if n_jobs == 1:
        # Serial processing with progress bar
        for item in tqdm(iterable, desc=desc, total=total):
            results.append(function(item))
    else:
        # Parallel processing with progress bar using map
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Using map with chunksize to optimize task submission
            for result in tqdm(executor.map(function, iterable, chunksize=chunksize), desc=desc, total=total):
                results.append(result)
    return results

def rasterise_single_tile(args):
    """
    Rasterize a single Voronoi region into grid tiles.

    Parameters:
    - args: Tuple containing (region, point, vor).

    Returns:
    - tile_df: DataFrame containing rasterized tile information.
    """
    try:
        region, point, index, vor = args
        vertices = vor.vertices[region]
        polygon = Polygon(vertices)

        # Compute bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Generate grid points within the bounding box
        x_range = np.arange(int(np.floor(minx)), int(np.ceil(maxx)) + 1)
        y_range = np.arange(int(np.floor(miny)), int(np.ceil(maxy)) + 1)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        # Vectorized point-in-polygon test using shapely.vectorized.contains
        mask = contains(polygon, grid_points[:, 0], grid_points[:, 1])
        pixels = grid_points[mask]

        if len(pixels) == 0:
            # If no pixels are inside the polygon, return an empty DataFrame
            return pd.DataFrame()

        # Calculate origin flags using vectorized operations
        origin_x = int(np.round(point[0]))
        origin_y = int(np.round(point[1]))
        origins = ((pixels[:, 0] == origin_x) & (pixels[:, 1] == origin_y)).astype(int)

        # Create DataFrame for the rasterized tile
        tile_df = pd.DataFrame({
            'x': pixels[:, 0],
            'y': pixels[:, 1],
            'barcode': str(index),
            'origin': origins
        })

        return tile_df

    except Exception as e:
        # Log the error with relevant information
        logging.error(f"Error processing tile with point {args[1]}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed

def process_counts(
    adata,
    method='log_norm',
    use_counts='raw',
    nfeatures=2000,
    min_cutoff='q5',
    verbose=True
):
    """
    Process count data with specified normalization and feature selection.

    Parameters:
    - adata: AnnData object containing count data.
    - method: Normalization method ('log_norm', 'SCT', 'TFIDF', 'none').
    - use_counts: Which counts to use ('raw' or layer name).
    - nfeatures: Number of highly variable genes to select.
    - min_cutoff: Minimum cutoff for feature selection.
    - verbose: Whether to print verbose messages.

    Returns:
    - adata_proc: Processed AnnData object.
    """
    if verbose:
        print(f"Processing counts with method: {method}")

    # Select counts based on 'use_counts' parameter
    if use_counts == 'raw':
        counts = adata.raw.X if adata.raw is not None else adata.X.copy()
        adata_proc = adata.copy()
    else:
        counts = adata.layers[use_counts]
        adata_proc = AnnData(counts)
        adata_proc.var_names = adata.var_names.copy()
        adata_proc.obs_names = adata.obs_names.copy()

    # Apply normalization and feature selection
    if method == 'log_norm':
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
        sc.pp.log1p(adata_proc)
        sc.pp.highly_variable_genes(adata_proc, n_top_genes=nfeatures)
        # adata_proc = adata_proc[:, adata_proc.var['highly_variable']]
    elif method == 'SCT':
        raise NotImplementedError("SCTransform normalization is not implemented in this code.")
    elif method == 'TFIDF':
        # Calculate Term Frequency (TF)
        tf = counts / counts.sum(axis=1)
        # Calculate Inverse Document Frequency (IDF)
        idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
        counts_tfidf = tf.multiply(idf)
        adata_proc.X = counts_tfidf

        # Feature selection based on variance
        if min_cutoff.startswith('q'):
            quantile = float(min_cutoff[1:]) / 100
            if scipy.sparse.issparse(counts_tfidf):
                counts_tfidf_dense = counts_tfidf.toarray()
            else:
                counts_tfidf_dense = counts_tfidf
            variances = np.var(counts_tfidf_dense, axis=0)
            cutoff = np.quantile(variances, quantile)
            selected_features = variances >= cutoff
            adata_proc = adata_proc[:, selected_features]
        else:
            raise ValueError("Invalid min_cutoff format.")
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")

    if verbose:
        print("Counts processing completed.")

    return adata_proc

def embed_latent_space(
    adata_proc,
    dim_reduction='PCA',
    dimensions=30,
    features=None,
    remove_lsi_1=True,
    verbose=True,
    n_jobs=None
):
    """
    Embed the processed data into a latent space using specified dimensionality reduction.

    Parameters:
    - adata_proc: Processed AnnData object.
    - dim_reduction: Method for dimensionality reduction ('PCA', 'UMAP', etc.).
    - dimensions: Number of dimensions for reduction.
    - features: Specific features to use.
    - remove_lsi_1: Whether to remove the first LSI component.
    - verbose: Whether to print verbose messages.
    - n_jobs: Number of parallel jobs.

    Returns:
    - embeds: Numpy array of embedded coordinates.
    """
    if verbose:
        print(f"Embedding latent space using {dim_reduction}...")

    # Subset to specific features if provided
    if features is not None:
        adata_proc = adata_proc[:, features]

    embeds = None

    if dim_reduction == 'PCA':
        sc.tl.pca(adata_proc, n_comps=dimensions,)
        embeds = adata_proc.obsm['X_pca']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'PCA_L':
        sc.tl.pca(adata_proc, n_comps=dimensions)
        loadings = adata_proc.varm['PCs']
        counts = adata_proc.X

        # Create iterable of arguments for PCA loadings calculation
        iterable = zip(range(counts.shape[0]), itertools.repeat(counts), itertools.repeat(loadings))
        desc = "Calculating PCA loadings"

        # Process PCA loadings in parallel or serially using the optimized helper function
        embeds_list = _process_in_parallel_map(
            calculate_pca_loadings,
            iterable,
            desc,
            n_jobs,
            chunksize=1000,
            total=counts.shape[0]
        )
        embeds = np.array(embeds_list)
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'UMAP':
        sc.pp.neighbors(adata_proc, n_pcs=dimensions)
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'LSI':
        embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
    elif dim_reduction == 'LSI_UMAP':
        lsi_embeds = run_lsi(adata_proc, n_components=dimensions, remove_first=remove_lsi_1)
        adata_proc.obsm['X_lsi'] = lsi_embeds
        sc.pp.neighbors(adata_proc, use_rep='X_lsi')
        sc.tl.umap(adata_proc, n_components=3)
        embeds = adata_proc.obsm['X_umap']
        embeds = MinMaxScaler().fit_transform(embeds)
    elif dim_reduction == 'NMF':
        nmf_model = NMF(n_components=dimensions, init='random', random_state=0)
        W = nmf_model.fit_transform(adata_proc.X)
        embeds = MinMaxScaler().fit_transform(W)
    else:
        raise ValueError(f"Dimensionality reduction method '{dim_reduction}' is not recognized.")

    if verbose:
        print("Latent space embedding completed.")

    return embeds

def calculate_pca_loadings(args):
    """
    Calculate PCA loadings for a single cell.

    Parameters:
    - args: Tuple containing (cell index, counts matrix, loadings matrix).

    Returns:
    - cell_embedding: Numpy array of PCA loadings for the cell.
    """
    i, counts, loadings = args
    if scipy.sparse.issparse(counts):
        expressed_genes = counts[i].toarray().flatten() > 0
    else:
        expressed_genes = counts[i] > 0
    cell_loadings = np.abs(loadings[expressed_genes, :])
    cell_embedding = cell_loadings.sum(axis=0)
    return cell_embedding

def run_lsi(adata, n_components=30, remove_first=True):
    """
    Run Latent Semantic Indexing (LSI) on the data.

    Parameters:
    - adata: AnnData object containing count data.
    - n_components: Number of LSI components.
    - remove_first: Whether to remove the first LSI component.

    Returns:
    - embeddings: Numpy array of LSI embeddings.
    """
    counts = adata.X
    if scipy.sparse.issparse(counts):
        counts = counts.tocsc()
        tf = counts.multiply(1 / counts.sum(axis=1))
    else:
        tf = counts / counts.sum(axis=1)[:, None]
    idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
    if scipy.sparse.issparse(counts):
        tfidf = tf.multiply(idf)
    else:
        tfidf = tf * idf

    from sklearn.utils.extmath import randomized_svd
    U, Sigma, VT = randomized_svd(tfidf, n_components=n_components + 1)

    if remove_first:
        U = U[:, 1:]
        Sigma = Sigma[1:]
        VT = VT[1:, :]

    embeddings = np.dot(U, np.diag(Sigma))
    embeddings = MinMaxScaler().fit_transform(embeddings)

    return embeddings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def image_plot(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    figsize=(10, 10),
    point_size=None,  # Set default to None
    scaling_factor = 37,
    origin=True,
):
    """
    Visualize embeddings as an image.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings and spatial coordinates.
    dimensions : list of int
        List of dimensions to use for visualization (1 or 3 dimensions).
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    figsize : tuple of float
        Figure size in inches (width, height).
    point_size : float or None
        Size of the points in the scatter plot. If None, it will be automatically determined based on figsize.
    """
    if len(dimensions) not in [1, 3]:
        raise ValueError("Only 1 or 3 dimensions can be used for visualization.")
    # Extract embeddings
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    # Extract spatial coordinates
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']
    
    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    # Normalize embeddings between 0 and 1
    coordinates = rebalance_colors(coordinates_df, dimensions)

    if len(dimensions) == 3:
        # Normalize the RGB values between 0 and 1
        cols = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b in zip(coordinates['R'], coordinates['G'], coordinates['B'])
        ]
    else:
        cols = np.repeat(coordinates['Grey'].values[:, np.newaxis], 3, axis=1) / 255.0  # Repeat gray values across RGB channels
        cols = cols.tolist()

    # Automatically determine point_size if not provided
    if point_size is None:

        # Adjust the scaling factor (e.g., 100) as needed to achieve desired overlap
        figure_area = figsize[0] * figsize[1]  # in square inches
        point_size = scaling_factor * (figure_area / 100)  # Adjust denominator as needed

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    sc_kwargs = dict(
        x=coordinates['x'],
        y=coordinates['y'],
        s=point_size,
        c=cols,
        marker='s',
        linewidths=0
    )
    scatter = ax.scatter(**sc_kwargs)
    ax.set_aspect('equal')
    ax.axis('off')
    title = f"{embedding} - Dims = {', '.join(map(str, dimensions))}"
    ax.set_title(title, fontsize=figsize[0] * 1.5)

    # Display the figure
    plt.show()
    # Do not return the figure to prevent automatic display

def min_max(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))

def rebalance_colors(coordinates, dimensions, method="minmax"):
    if len(dimensions) == 3:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]]
        colors = coordinates.iloc[:, [4, 5, 6]].values
        
        if method == "minmax":
            colors = np.apply_along_axis(min_max, 0, colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.DataFrame(colors, columns=["R", "G", "B"])], axis=1)
    
    else:
        template = coordinates.loc[:, ["barcode", "x", "y", "origin"]]
        colors = coordinates.iloc[:, 4].values
        
        if method == "minmax":
            colors = min_max(colors)
        else:
            colors[colors < 0] = 0
            colors[colors > 1] = 1
        
        template = pd.concat([template, pd.Series(colors, name="Grey")], axis=1)
    
    return template


# def rebalance_colors(embeddings, method='minmax'):
#     """
#     Normalize embeddings between 0 and 1.
    
#     Parameters
#     ----------
#     embeddings : numpy.ndarray
#         Embedding matrix.
#     method : str
#         Normalization method ('minmax' or 'truncate').
    
#     Returns
#     -------
#     embeddings_norm : numpy.ndarray
#         Normalized embeddings.
#     """
#     if method == 'minmax':
#         min_vals = embeddings.min(axis=0)
#         max_vals = embeddings.max(axis=0)
#         embeddings_norm = (embeddings - min_vals) / (max_vals - min_vals + 1e-8)
#     elif method == 'truncate':
#         embeddings_norm = np.clip(embeddings, 0, 1)
#     else:
#         raise ValueError("Normalization method must be 'minmax' or 'truncate'.")
#     return embeddings_norm
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def territory_plot(
    adata,
    trial='last',
    split=False,
    highlight=None,
    contour='None',
    randomize=True,
    cex=10,
    cex_pt=1,
    alpha=0.65,
    use_image=False
):
    """
    Plot territories (clusters) on spatial coordinates.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data and territories.
    trial : str
        The key in adata.obs where territory information is stored.
    split : bool
        Whether to split territories into separate subplots.
    highlight : list of int or None
        List of territory labels to highlight.
    contour : str
        Contour method ('None', 'convex', 'concave').
    randomize : bool
        Whether to randomize colors.
    cex : float
        Font size scaling factor.
    cex_pt : float
        Point size scaling factor.
    alpha : float
        Transparency of points.
    use_image : bool
        Whether to use background image.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Check if territories are available
    if trial not in adata.obs:
        raise ValueError(f"Territory information '{trial}' not found in adata.obs.")

    # Get spatial coordinates and territory labels
    coordinates = adata.obsm['spatial']
    territories = adata.obs[trial].astype(str)

    # Create color palette
    unique_territories = territories.unique()
    num_territories = len(unique_territories)
    colors = create_palette(num_territories, randomize)
    color_map = dict(zip(unique_territories, colors))
    territory_colors = territories.map(color_map)

    # Adjust transparency if highlighting
    if highlight is not None:
        highlight = set(map(str, highlight))
        alphas = territories.apply(lambda x: alpha if x in highlight else alpha * 0.25)
    else:
        alphas = alpha

    # Plotting
    if split:
        n_cols = min(4, num_territories)
        n_rows = int(np.ceil(num_territories / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(cex * n_cols, cex * n_rows))
        axes = axes.flatten()
        for idx, territory_label in enumerate(unique_territories):
            ax = axes[idx]
            mask = territories == territory_label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=territory_colors[mask],
                s=cex_pt,
                alpha=alphas[mask],
                label=territory_label
            )
            ax.set_title(f"Territory {territory_label}")
            ax.set_aspect('equal')
            ax.axis('off')
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(cex, cex))
        sc_kwargs = dict(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            s=cex_pt,
            c=territory_colors,
            alpha=alphas,
            linewidths=0
        )
        ax.scatter(**sc_kwargs)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Territories ({trial})", fontsize=cex * 1.5)
        # Create legend
        handles = [patches.Patch(color=color_map[t], label=t) for t in unique_territories]
        ax.legend(handles=handles, fontsize=cex, title='Territory')
    plt.show()
    return fig
def create_palette(num_colors, randomize=True):
    """
    Create a color palette.
    
    Parameters
    ----------
    num_colors : int
        Number of colors needed.
    randomize : bool
        Whether to randomize the colors.
    
    Returns
    -------
    colors : list
        List of color hex codes.
    """
    import seaborn as sns
    base_colors = sns.color_palette('tab10', n_colors=10)
    if num_colors <= 10:
        colors = base_colors[:num_colors]
    else:
        colors = sns.color_palette('hsv', n_colors=num_colors)
    if randomize:
        np.random.shuffle(colors)
    return colors
def view_gene_expression(
    adata,
    genes,
    norm_method='last',
    trial='last',
    territory_1=None,
    territory_2=None,
    cells=None,
    norm=True,
    as_layer=False,
    cex=10,
    cex_pt=1,
    alpha=0.75,
    max_size=5,
    return_as_list=False
):
    """
    Visualize gene expression on spatial coordinates.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data and expression values.
    genes : list of str
        List of genes to visualize.
    norm_method : str
        Normalization method to use.
    trial : str
        The key in adata.obs where territory information is stored.
    territory_1 : list or None
        List of territories in group 1.
    territory_2 : list or None
        List of territories in group 2.
    cells : list or None
        List of cell barcodes to highlight.
    norm : bool
        Whether to normalize expression values.
    as_layer : bool
        Whether to average expression in territories.
    cex : float
        Font size scaling factor.
    cex_pt : float
        Point size scaling factor.
    alpha : float
        Transparency of points.
    max_size : int
        Maximum size for subplot grid.
    return_as_list : bool
        Whether to return the plots as a list.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or list of Figures
        The generated figure(s).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if not isinstance(genes, list):
        genes = [genes]
    
    plots = []
    for gene in genes:
        if gene not in adata.var_names:
            print(f"Warning: Gene '{gene}' not found in adata.var_names.")
            continue

        # Get expression values
        expr = adata[:, gene].X
        if scipy.sparse.issparse(expr):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()

        # Normalize expression
        if norm:
            expr = (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)

        # Filter territories if specified
        if trial in adata.obs:
            territories = adata.obs[trial].astype(str)
        else:
            territories = pd.Series(['All'] * adata.n_obs, index=adata.obs_names)
        mask = pd.Series(True, index=adata.obs_names)
        if territory_1 is not None:
            territory_1 = set(map(str, territory_1))
            mask &= territories.isin(territory_1)
        if territory_2 is not None:
            territory_2 = set(map(str, territory_2))
            mask &= territories.isin(territory_2)
        if cells is not None:
            mask &= adata.obs_names.isin(cells)

        # Prepare data
        coordinates = adata.obsm['spatial'][mask.values]
        expr_filtered = expr[mask.values]
        territories_filtered = territories[mask.values]

        # Plotting
        fig, ax = plt.subplots(figsize=(cex, cex))
        sc_kwargs = dict(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            s=cex_pt * 10,
            c=expr_filtered,
            cmap='Spectral_r',
            alpha=alpha,
            linewidths=0
        )
        scatter = ax.scatter(**sc_kwargs)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Gene Expression: {gene}", fontsize=cex * 1.5)
        # Add colorbar
        norm = Normalize(vmin=expr_filtered.min(), vmax=expr_filtered.max())
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='Spectral_r'), ax=ax)
        cbar.set_label('Expression' if not norm else 'Normalized Expression', fontsize=cex)
        plots.append(fig)

    if len(plots) == 1:
        return plots[0]
    elif return_as_list:
        return plots
    else:
        # Arrange plots in grid
        n_cols = min(max_size, int(np.ceil(np.sqrt(len(plots)))))
        n_rows = int(np.ceil(len(plots) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(cex * n_cols, cex * n_rows))
        axes = axes.flatten()
        for ax, plot_fig in zip(axes, plots):
            plot_ax = plot_fig.axes[0]
            ax.imshow(plot_ax.images[0].get_array(), extent=plot_ax.images[0].get_extent())
            ax.set_title(plot_ax.get_title(), fontsize=cex * 1.5)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        return fig


import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.filters import median
from skimage.morphology import disk
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from typing import List, Union, Optional
import warnings

def smooth_image(
    adata,
    dimensions: List[int] = [0, 1, 2],
    embedding: str = 'X_embedding',
    method: Union[str, List[str]] = 'iso',
    iter: int = 1,
    sigma: Union[float, List[float]] = 1,
    box: Union[int, List[int]] = 20,
    threshold: float = 0,
    neuman: bool = True,
    gaussian_flag: bool = True,  # 'gaussian' is a reserved keyword in some contexts
    na_rm: bool = False,
    across_levels: str = "min",
    origin: bool = True,
    verbose: bool = True
):
    """
    Apply iterative smoothing to embeddings in AnnData object, mirroring the R 'smooth_image' function.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the embeddings to smooth.
    dimensions : list of int, optional (default: [0, 1, 2])
        Latent space dimensions to use for smoothing.
    embedding : str, optional (default: 'X_embedding')
        Name of the embedding in `adata.obsm` to smooth.
    method : str or list of str, optional (default: 'iso')
        Smoothing method(s) to apply. Options: "median", "iso", "box".
    iter : int, optional (default: 1)
        Number of smoothing iterations.
    sigma : float or list of float, optional (default: 1)
        Standard deviation for Gaussian (iso) smoothing.
    box : int or list of int, optional (default: 3)
        Size for box or median smoothing.
    threshold : float, optional (default: 0)
        Discard pixels below this value (applicable to box/median methods).
    neuman : bool, optional (default: True)
        Use Neumann boundary conditions if True, else Dirichlet.
    gaussian_flag : bool, optional (default: True)
        Use Gaussian filter if True.
    na_rm : bool, optional (default: False)
        Remove or ignore NA values during smoothing.
    across_levels : str, optional (default: "min")
        Aggregation method after applying multiple smoothing levels. Options: "min", "mean", "max".
    origin : bool, optional (default: True)
        If True, use tiles with origin == 1; else use all tiles.
    verbose : bool, optional (default: True)
        If True, display progress messages.

    Returns
    -------
    adata : AnnData
        The AnnData object with smoothed embeddings added to `adata.obsm` as 'X_embedding_smooth'.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting smoothing...")

    # Ensure method is a list
    methods = method if isinstance(method, list) else [method]
    # Ensure sigma and box are lists for multiple levels
    sigmas = sigma if isinstance(sigma, list) else [sigma]
    boxes = box if isinstance(box, list) else [box]

    # Validate across_levels
    if across_levels not in ["min", "mean", "max"]:
        raise ValueError("across_levels must be one of 'min', 'mean', 'max'")

    # -------------------------------------------------------------------------- #
    # Shifting format as per user-provided code
    # -------------------------------------------------------------------------- #
    if origin:
        tiles = adata.uns['tiles'][adata.uns['tiles']['origin'] == 1]
    else:
        tiles = adata.uns['tiles']

    tile_colors = pd.DataFrame(np.array(adata.obsm[embedding])[:, dimensions])
    tile_colors['barcode'] = adata.obs.index

    # Merge tiles with embeddings based on 'barcode'
    coordinates_df = pd.merge(tiles, tile_colors, on="barcode", how="right").dropna().reset_index(drop=True)

    # Extract spatial coordinates and embeddings
    spatial_coords = coordinates_df.loc[:, ["barcode", "x", "y", "origin"]]
    embeddings = coordinates_df.drop(columns=["barcode", "x", "y", "origin"]).values  # Convert to NumPy array

    x = spatial_coords['x'].values
    y = spatial_coords['y'].values

    # Create a grid mapping
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Define grid resolution
    grid_size = 1  # Adjust as needed
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Map coordinates to grid indices
    x_idx = np.digitize(x, bins=x_bins) - 1
    y_idx = np.digitize(y, bins=y_bins) - 1

    # Initialize grid arrays for each dimension
    grid_shape = (len(y_bins), len(x_bins))
    embedding_grids = [np.full(grid_shape, np.nan) for _ in range(len(dimensions))]

    # Place embedding values into the grids
    for i, dim in enumerate(range(len(dimensions))):
        embedding_grid = embedding_grids[i]
        embedding_values = embeddings[:, i]
        embedding_grid[y_idx, x_idx] = embedding_values
        embedding_grids[i] = embedding_grid

    # Apply smoothing to each grid with multiple levels
    for iter_num in range(iter):
        if verbose:
            print(f"Iteration {iter_num + 1}/{iter}")

        for i, dim in enumerate(range(len(dimensions))):
            embedding_grid = embedding_grids[i]
            smoothed_grids = []

            # Apply each combination of method, sigma, box
            for m in methods:
                if m == "iso":
                    for s in sigmas:
                        smoothed = internal_smooth(
                            embedding_grid,
                            method=m,
                            sigma=s,
                            size=None,  # Size not used for 'iso'
                            threshold=threshold,
                            neuman=neuman,
                            gaussian=gaussian_flag,
                            na_rm=na_rm
                        )
                        smoothed_grids.append(smoothed)
                elif m in ["median", "box"]:
                    for b in boxes:
                        smoothed = internal_smooth(
                            embedding_grid,
                            method=m,
                            sigma=None,  # Sigma not used for 'median' or 'box'
                            size=b,
                            threshold=threshold,
                            neuman=neuman,
                            gaussian=gaussian_flag,
                            na_rm=na_rm
                        )
                        smoothed_grids.append(smoothed)
                else:
                    raise ValueError(f"Unknown smoothing method '{m}'")

            # Aggregate across levels
            if across_levels == "min":
                aggregated = np.nanmin(np.stack(smoothed_grids), axis=0)
            elif across_levels == "mean":
                aggregated = np.nanmean(np.stack(smoothed_grids), axis=0)
            elif across_levels == "max":
                aggregated = np.nanmax(np.stack(smoothed_grids), axis=0)

            # Restore NaNs
            aggregated[np.isnan(embedding_grid)] = np.nan
            embedding_grids[i] = aggregated

    # Map the smoothed grid values back to embeddings
    for i, dim in enumerate(range(len(dimensions))):
        embedding_grid = embedding_grids[i]
        embeddings[:, i] = embedding_grid[y_idx, x_idx]

    # Update the embeddings in adata
    if 'X_embedding_smooth' not in adata.obsm:
        adata.obsm['X_embedding_smooth'] = np.full((adata.n_obs, len(dimensions)), np.nan)

    # Assign smoothed embeddings
    adata.obsm['X_embedding_smooth'][:, dimensions] = embeddings

    if verbose:
        print("Smoothing completed.")

    return adata


def internal_smooth(
    embedding_grid: np.ndarray,
    method: str = 'iso',
    sigma: Optional[float] = None,
    size: Optional[int] = None,
    threshold: float = 0,
    neuman: bool = True,
    gaussian: bool = True,
    na_rm: bool = False
) -> np.ndarray:
    """
    Internal function to perform smoothing on a 2D grid.

    Parameters
    ----------
    embedding_grid : np.ndarray
        2D array representing the embedding grid.
    method : str, optional (default: 'iso')
        Smoothing method: "median", "iso", or "box".
    sigma : float, optional
        Standard deviation for Gaussian (iso) smoothing.
    size : int, optional
        Size for box or median smoothing.
    threshold : float, optional (default: 0)
        Discard pixels below this value (applicable to box/median methods).
    neuman : bool, optional (default: True)
        Use Neumann boundary conditions if True, else Dirichlet.
    gaussian : bool, optional (default: True)
        Use Gaussian filter if True.
    na_rm : bool, optional (default: False)
        Remove or ignore NA values during smoothing.

    Returns
    -------
    embedding_grid_smoothed : np.ndarray
        Smoothed 2D array.
    """
    if np.isnan(embedding_grid).all():
        return embedding_grid

    # Handle boundary conditions
    mode = 'nearest' if neuman else 'constant'

    # Replace NaNs with the mean or zero
    if na_rm:
        mean_val = np.nanmean(embedding_grid)
        embedding_grid_filled = np.where(np.isnan(embedding_grid), mean_val, embedding_grid)
    else:
        embedding_grid_filled = np.copy(embedding_grid)
        embedding_grid_filled[np.isnan(embedding_grid)] = 0

    if method == 'median':
        if size is None:
            raise ValueError("Size must be specified for median filtering.")
        selem = disk(size)
        embedding_grid_smoothed = median(embedding_grid_filled, selem=selem)
    elif method == 'iso':
        if sigma is None:
            raise ValueError("Sigma must be specified for isoblur.")
        embedding_grid_smoothed = gaussian_filter(embedding_grid_filled, sigma=sigma, mode=mode)
    elif method == 'box':
        if size is None:
            raise ValueError("Size must be specified for box filtering.")
        embedding_grid_smoothed = uniform_filter(embedding_grid_filled, size=size, mode=mode)
    else:
        raise ValueError(f"Unknown smoothing method '{method}'")

    # Apply threshold if applicable
    if method in ['median', 'box'] and threshold > 0:
        embedding_grid_smoothed = np.where(embedding_grid_smoothed >= threshold, embedding_grid_smoothed, np.nan)

    # Restore NaNs
    embedding_grid_smoothed[np.isnan(embedding_grid)] = np.nan

    return embedding_grid_smoothed



from sklearn.neighbors import NearestNeighbors

def smooth_image_knn(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    iterations=1,
    k=5,
    verbose=True
):
    """
    Apply KNN smoothing to embeddings in AnnData object.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting KNN smoothing...")

    embeddings = adata.obsm[embedding][:, dimensions].copy()
    spatial_coords = adata.obsm['spatial']

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    for iter_num in range(iterations):
        if verbose:
            print(f"Iteration {iter_num + 1}/{iterations}")
        embeddings_new = np.zeros_like(embeddings)
        for i in range(embeddings.shape[0]):
            neighbor_indices = indices[i]
            embeddings_new[i] = embeddings[neighbor_indices].mean(axis=0)
        embeddings = embeddings_new

    # Update the embeddings in adata
    adata.obsm[embedding][:, dimensions] = embeddings

    if verbose:
        print("KNN smoothing completed.")

import numpy as np
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from anndata import AnnData
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from typing import List
import warnings



def equalize_image(
    adata: AnnData,
    dimensions: List[int] = [0, 1, 2],
    embedding: str = 'X_embedding',
    method: str = 'BalanceSimplest',
    N: int = 1,
    smax: float = 1.0,
    sleft: float = 1.0,
    sright: float = 1.0,
    lambda_: float = 0.1,
    up: float = 100.0,
    down: float = 10.0,
    verbose: bool = True,
    n_jobs: int = 1  # 병렬 처리 스레드 수
) -> AnnData:
    """
    Equalize histogram of embeddings.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for equalization (0-based indexing).
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    method : str
        Equalization method: 'BalanceSimplest', 'EqualizePiecewise', 'SPE', 
        'EqualizeDP', 'EqualizeADP', 'ECDF', 'histogram', 'adaptive'.
    N : int, optional
        Number of segments for EqualizePiecewise (default is 1).
    smax : float, optional
        Upper limit for contrast stretching in EqualizePiecewise.
    sleft : float, optional
        Percentage of pixels to saturate on the left side for BalanceSimplest (default is 1.0).
    sright : float, optional
        Percentage of pixels to saturate on the right side for BalanceSimplest (default is 1.0).
    lambda_ : float, optional
        Strength of background correction for SPE (default is 0.1).
    up : float, optional
        Upper color value threshold for EqualizeDP (default is 100.0).
    down : float, optional
        Lower color value threshold for EqualizeDP (default is 10.0).
    verbose : bool, optional
        Whether to display progress messages (default is True).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    
    Returns
    -------
    AnnData
        The updated AnnData object with equalized embeddings.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")
    
    if verbose:
        print("Starting histogram equalization...")
    
    embeddings = adata.obsm[embedding][:, dimensions].copy()
    
    def process_dimension(i, dim):
        if verbose:
            print(f"Equalizing dimension {dim} using method '{method}'")
        data = embeddings[:, i]
        
        if method == 'BalanceSimplest':
            return balance_simplest(data, sleft=sleft, sright=sright)
        elif method == 'EqualizePiecewise':
            return equalize_piecewise(data, N=N, smax=smax)
        elif method == 'SPE':
            return spe_equalization(data, lambda_=lambda_)
        elif method == 'EqualizeDP':
            return equalize_dp(data, down=down, up=up)
        elif method == 'EqualizeADP':
            return equalize_adp(data)
        elif method == 'ECDF':
            return ecdf_eq(data)
        elif method == 'histogram':
            return equalize_hist(data)
        elif method == 'adaptive':
            return equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
        else:
            raise ValueError(f"Unknown equalization method '{method}'")
    
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_dimension, range(len(dimensions)), dimensions))
    else:
        results = [process_dimension(i, dim) for i, dim in enumerate(dimensions)]
    
    for i, result in enumerate(results):
        embeddings[:, i] = result
    # Update the embeddings in adata
    if 'X_embedding_equalize' not in adata.obsm:
        adata.obsm['X_embedding_equalize'] = np.full((adata.n_obs, len(dimensions)), np.nan)
    # Update the embeddings in adata
    adata.obsm['X_embedding_equalize'][:, dimensions] = embeddings
    
    # 로그 기록
    if verbose:
        print("Logging changes to AnnData.uns['equalize_image_log']")
    adata.uns['equalize_image_log'] = {
        'method': method,
        'parameters': {
            'N': N,
            'smax': smax,
            'sleft': sleft,
            'sright': sright,
            'lambda_': lambda_,
            'up': up,
            'down': down
        },
        'dimensions': dimensions,
        'embedding': embedding
    }
    
    if verbose:
        print("Histogram equalization completed.")
    
    return adata


# Placeholder implementations for methods not available in skimage
def balance_simplest(data: np.ndarray, sleft: float = 1.0, sright: float = 1.0, range_limits: tuple = (0, 1)) -> np.ndarray:
    """
    BalanceSimplest equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    sleft : float
        Percentage of pixels to saturate on the left.
    sright : float
        Percentage of pixels to saturate on the right.
    range_limits : tuple
        Output range after rescaling.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    p_left = np.percentile(data, sleft)
    p_right = np.percentile(data, 100 - sright)
    data_eq = rescale_intensity(data, in_range=(p_left, p_right), out_range=range_limits)
    return data_eq

def equalize_piecewise(data: np.ndarray, N: int = 1, smax: float = 1.0) -> np.ndarray:
    """
    EqualizePiecewise equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    N : int
        Number of segments to divide the data into.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    data_eq = np.copy(data)
    quantiles = np.linspace(0, 100, N + 1)
    for i in range(N):
        lower = np.percentile(data, quantiles[i])
        upper = np.percentile(data, quantiles[i + 1])
        mask = (data >= lower) & (data < upper)
        if np.any(mask):
            segment = rescale_intensity(data[mask], in_range=(lower, upper), out_range=(0, 1))
            data_eq[mask] = np.minimum(segment, smax)  # smax 적용
    return data_eq

def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    """
    SPE (Screened Poisson Equation) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    lambda_ : float
        Strength parameter for background correction.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    # Placeholder implementation: apply Gaussian smoothing
    data_smoothed = gaussian_filter(data, sigma=lambda_)
    data_eq = rescale_intensity(data_smoothed, out_range=(0, 1))
    return data_eq

# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve

# def spe_equalization(data: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
#     data_2d = data.reshape(1, -1)
#     n = data_2d.shape[1]
#     diagonals = [diags([1 + 2*lambda_], [0], shape=(n, n))]
#     A = diags([-lambda_, -lambda_], [-1, 1], shape=(n, n))
#     A = A + diagonals[0]
#     b = data_2d.flatten()
#     data_eq = spsolve(A, b)
#     return np.clip(data_eq, 0, 1)




def equalize_dp(data: np.ndarray, down: float = 10.0, up: float = 100.0) -> np.ndarray:
    """
    EqualizeDP (Dynamic Programming) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    down : float
        Lower percentile threshold.
    up : float
        Upper percentile threshold.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    p_down = np.percentile(data, down)
    p_up = np.percentile(data, up)
    data_clipped = np.clip(data, p_down, p_up)
    data_eq = rescale_intensity(data_clipped, out_range=(0, 1))
    return data_eq

def equalize_adp(data: np.ndarray) -> np.ndarray:
    """
    EqualizeADP (Adaptive Dynamic Programming) equalization.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    # Placeholder implementation: use adaptive histogram equalization
    data_eq = equalize_adapthist(data.reshape(1, -1), clip_limit=0.03).flatten()
    return data_eq

def ecdf_eq(data: np.ndarray) -> np.ndarray:
    """
    Equalize data using empirical cumulative distribution function (ECDF).
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data to equalize.
    
    Returns
    -------
    np.ndarray
        Equalized data.
    """
    sorted_idx = np.argsort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    data_eq = np.zeros_like(data)
    data_eq[sorted_idx] = ecdf
    return data_eq

import numpy as np
from skimage.restoration import denoise_tv_chambolle

def regularise_image(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    weight=0.1,
    n_iter_max=200,
    grid_size=None,
    verbose=True
):
    """
    Denoise embeddings via total variation regularization.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting image regularization...")

    embeddings = adata.obsm[embedding][:, dimensions].copy()

    # Get spatial coordinates
    spatial_coords = adata.obsm['spatial']
    x = spatial_coords[:, 0]
    y = spatial_coords[:, 1]

    # Create a grid mapping
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if grid_size is None:
        # Calculate grid size based on data density
        x_range = x_max - x_min
        y_range = y_max - y_min
        grid_size = max(x_range, y_range) / 100  # Adjust 100 as needed

    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Map coordinates to grid indices
    x_idx = np.digitize(x, bins=x_bins) - 1
    y_idx = np.digitize(y, bins=y_bins) - 1

    # Initialize grid arrays for each dimension
    grid_shape = (len(y_bins), len(x_bins))
    embedding_grids = [np.full(grid_shape, np.nan) for _ in dimensions]

    # Place embedding values into the grids
    for i, dim in enumerate(dimensions):
        embedding_grid = embedding_grids[i]
        embedding_values = embeddings[:, i]
        embedding_grid[y_idx, x_idx] = embedding_values

    # Apply regularization to each grid
    for i in range(len(dimensions)):
        embedding_grid = embedding_grids[i]
        embedding_grid = regularise(embedding_grid, weight=weight, n_iter_max=n_iter_max)
        embedding_grids[i] = embedding_grid

    # Map the regularized grid values back to embeddings
    for i in range(len(dimensions)):
        embedding_grid = embedding_grids[i]
        embeddings[:, i] = embedding_grid[y_idx, x_idx]

    # Update the embeddings in adata
    adata.obsm[embedding][:, dimensions] = embeddings

    if verbose:
        print("Image regularization completed.")

def regularise(embedding_grid, weight=0.1, n_iter_max=200):
    """
    Denoise data using total variation regularization.
    """
    # Handle NaNs
    nan_mask = np.isnan(embedding_grid)
    if np.all(nan_mask):
        return embedding_grid  # Return as is if all values are NaN

    # Replace NaNs with mean of existing values
    mean_value = np.nanmean(embedding_grid)
    embedding_grid_filled = np.copy(embedding_grid)
    embedding_grid_filled[nan_mask] = mean_value

    # Apply total variation denoising
    # Adjust parameter based on scikit-image version
    from skimage import __version__ as skimage_version
    if skimage_version >= '0.19':
        # Use max_num_iter for scikit-image >= 0.19
        embedding_grid_denoised = denoise_tv_chambolle(
            embedding_grid_filled,
            weight=weight,
            max_num_iter=n_iter_max
        )
    else:
        # Use n_iter_max for scikit-image < 0.19
        embedding_grid_denoised = denoise_tv_chambolle(
            embedding_grid_filled,
            weight=weight,
            n_iter_max=n_iter_max
        )

    # Restore NaNs
    embedding_grid_denoised[nan_mask] = np.nan

    return embedding_grid_denoised
from sklearn.neighbors import NearestNeighbors

def regularise_image_knn(
    adata,
    dimensions=[0, 1, 2],
    embedding='X_embedding',
    weight=0.1,
    n_neighbors=5,
    n_iter=1,
    verbose=True
):
    """
    Denoise embeddings via KNN total variation regularization.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for denoising.
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    weight : float
        Denoising weight parameter.
    n_neighbors : int
        Number of neighbors to use in KNN.
    n_iter : int
        Number of iterations.
    verbose : bool
        Whether to display progress messages.
    
    Returns
    -------
    None
        The function updates the embeddings in adata.obsm[embedding].
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print("Starting KNN image regularization...")

    embeddings = adata.obsm[embedding][:, dimensions].copy()
    spatial_coords = adata.obsm['spatial']

    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    for iteration in range(n_iter):
        if verbose:
            print(f"Iteration {iteration + 1}/{n_iter}")
        embeddings_new = np.copy(embeddings)
        for i in range(embeddings.shape[0]):
            neighbor_indices = indices[i]
            neighbor_embeddings = embeddings[neighbor_indices]
            # Apply total variation denoising to the neighbors
            denoised_embedding = denoise_tv_chambolle(neighbor_embeddings, weight=weight)
            embeddings_new[i] = denoised_embedding[0]  # Update the current point
        embeddings = embeddings_new

    # Update the embeddings in adata
    adata.obsm[embedding][:, dimensions] = embeddings

    if verbose:
        print("KNN image regularization completed.")

from sklearn.cluster import KMeans
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import slic
from skimage import graph as skimage_graph
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import gray2rgb
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors


def segment_image(
    adata,
    dimensions=list(range(30)),
    embedding='X_embedding',
    method='slic',
    resolution=10.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42,
    Segment='Segment',
    verbose=True
):
    """
    Segment embeddings to find initial territories.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing embeddings.
    dimensions : list of int
        List of dimensions to use for segmentation.
    embedding : str
        Key in adata.obsm where the embeddings are stored.
    method : str
        Segmentation method: 'kmeans', 'louvain', 'leiden', 'slic', 'som', 'leiden_slic', 'louvain_slic'.
    resolution : float or int
        Resolution parameter for clustering methods.
    compactness : float
        Compactness parameter influencing the importance of spatial proximity in clustering.
    scaling : float
        Scaling factor for spatial coordinates.
    n_neighbors : int
        Number of neighbors for constructing the nearest neighbor graph.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to display progress messages.
    """
    if embedding not in adata.obsm:
        raise ValueError(f"Embedding '{embedding}' not found in adata.obsm.")

    if verbose:
        print(f"Starting image segmentation using method '{method}'...")

    embeddings = adata.obsm[embedding][:, dimensions]
    spatial_coords = adata.obsm['spatial']

    if method == 'kmeans':
        clusters = kmeans_segmentation(embeddings, resolution, random_state)
    elif method == 'louvain':
        clusters = louvain_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'leiden':
        clusters = leiden_segmentation(embeddings, resolution, n_neighbors, random_state)
    elif method == 'slic':
        clusters = slic_segmentation(adata, embeddings,spatial_coords, n_segments=int(resolution), compactness=compactness,scaling=scaling)
    elif method == 'som':
        clusters = som_segmentation(embeddings, resolution)
    elif method == 'leiden_slic':
        clusters = leiden_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    elif method == 'louvain_slic':
        clusters = louvain_slic_segmentation(embeddings, spatial_coords, resolution, compactness, scaling, n_neighbors, random_state)
    else:
        raise ValueError(f"Unknown segmentation method '{method}'")

    # Store the clusters in adata.obs
    adata.obs[Segment] = pd.Categorical(clusters)

    if verbose:
        print("Image segmentation completed.")


def louvain_segmentation(embeddings, resolution=1.0, n_neighbors = 1.5, random_state =42):
    """
    Perform Louvain clustering on embeddings.
    """
    import scanpy as sc

    # Create a temporary AnnData object
    temp_adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(temp_adata, n_neighbors = n_neighbors, use_rep='X')
    sc.tl.louvain(temp_adata, resolution=resolution, random_state = random_state)
    clusters = temp_adata.obs['louvain'].astype(int).values
    return clusters

def leiden_segmentation(embeddings, resolution=1.0, n_neighbors = 1.5, random_state =42):
    """
    Perform Leiden clustering on embeddings.
    """
    import scanpy as sc

    # Create a temporary AnnData object
    temp_adata = sc.AnnData(X=embeddings)
    sc.pp.neighbors(temp_adata, n_neighbors = n_neighbors, use_rep='X')
    sc.tl.leiden(temp_adata,flavor='igraph',n_iterations=2, resolution=resolution, random_state = random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values
    return clusters

def slic_segmentation(
    adata,
    embeddings,
    spatial_coords,
    n_segments=100,
    compactness=10,
    scaling=0.3,
    verbose=True
):
    """
    Perform SLIC superpixel segmentation on embeddings mapped to a 2D grid.
    """
    if verbose:
        print("Mapping embeddings onto a 2D grid based on spatial coordinates...")

    # Normalize spatial coordinates
    x = spatial_coords[:, 0]
    y = spatial_coords[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Define grid resolution
    grid_size = scaling  # Adjust this value as needed
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size]
    
    # Prepare the data for grid mapping
    points = np.vstack((x, y)).T
    grid_shape = grid_x.shape
    
    # Map each embedding dimension onto the grid
    embedding_grids = []
    for i in range(embeddings.shape[1]):
        embedding_values = embeddings[:, i]
        # Use griddata to interpolate embeddings onto the grid
        grid_z = griddata(points, embedding_values, (grid_x, grid_y), method='nearest')
        embedding_grids.append(grid_z)
    
    # Stack the embedding grids to create an image
    image = np.stack(embedding_grids, axis=-1)
    
    # Handle any NaN values in the image
    nan_mask = np.isnan(image)
    if np.any(nan_mask):
        image[nan_mask] = np.nanmean(image[~nan_mask])
    
    # Convert image to float type if necessary
    image = img_as_float(image)
    
    if verbose:
        print("Applying SLIC segmentation...")
    
    # Apply SLIC segmentation
    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,  # Updated argument
        convert2lab=False
    )
    
    # Map the segments back to the original data points
    # First, create a reverse mapping from grid coordinates to spatial coordinates
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    tree = NearestNeighbors(n_neighbors=1).fit(grid_points)
    distances, indices = tree.kneighbors(points)
    indices = indices.flatten()
    
    # Get the segment labels for each data point
    clusters = segments.reshape(-1)[indices]
    
    return clusters

def som_segmentation(embeddings, resolution=10):
    """
    Perform Self-Organizing Map (SOM) clustering.
    """
    som_grid_size = int(np.sqrt(resolution))
    som = MiniSom(som_grid_size, som_grid_size, embeddings.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(embeddings)
    som.train_random(embeddings, 100)
    win_map = som.win_map(embeddings)
    clusters = np.zeros(embeddings.shape[0], dtype=int)
    for i, x in enumerate(embeddings):
        winner = som.winner(x)
        clusters[i] = winner[0] * som_grid_size + winner[1]
    return clusters

def louvain_slic_segmentation(
    embeddings,
    spatial_coords,
    resolution=1.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42
):
    """
    Perform Louvain clustering on embeddings scaled with spatial coordinates and compactness.
    """
    # Compute scaling factors
    sc_spat = np.max([np.max(spatial_coords[:, 0]), np.max(spatial_coords[:, 1])]) * scaling
    sc_col = np.max(np.std(embeddings, axis=0))
    
    # Compute ratio
    ratio = (sc_spat / sc_col) / compactness
    
    # Scale embeddings
    embeddings_scaled = embeddings * ratio
    
    # Combine embeddings and spatial coordinates
    combined_data = np.concatenate([embeddings_scaled, spatial_coords], axis=1)
    
    # Create AnnData object
    temp_adata = sc.AnnData(X=combined_data)
    sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.louvain(temp_adata, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['louvain'].astype(int).values
    return clusters

def leiden_slic_segmentation(
    embeddings,
    spatial_coords,
    resolution=1.0,
    compactness=1.0,
    scaling=0.3,
    n_neighbors=15,
    random_state=42
):
    """
    Perform Leiden clustering on embeddings scaled with spatial coordinates and compactness.
    """
    # Compute scaling factors
    sc_spat = np.max([np.max(spatial_coords[:, 0]), np.max(spatial_coords[:, 1])]) * scaling
    sc_col = np.max(np.std(embeddings, axis=0))
    
    # Compute ratio
    ratio = (sc_spat / sc_col) / compactness
    
    # Scale embeddings
    embeddings_scaled = embeddings * ratio
    
    # Combine embeddings and spatial coordinates
    combined_data = np.concatenate([embeddings_scaled, spatial_coords], axis=1)
    
    # Create AnnData object
    temp_adata = sc.AnnData(X=combined_data)
    sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(temp_adata, flavor='igraph',n_iterations=2, resolution=resolution, random_state=random_state)
    clusters = temp_adata.obs['leiden'].astype(int).values
    return clusters
