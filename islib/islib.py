# Standard library imports
import base64
import io
import math
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Constants
DEFAULT_MAX_CLUSTERS = 50
DEFAULT_MIN_WINDOW = 10
DEFAULT_RESOLUTION = 100
DEFAULT_TOLERANCE = 0.1
DEFAULT_VARIANCE_THRESHOLD = 0.95
DEFAULT_PERCENTILE_LIMIT = 25
DEFAULT_MIN_SIZE_CLUSTER = 2
DEFAULT_SPIKE_THRESHOLD = 3
MIN_FREQUENCY_THRESHOLD = 1e-6


RESULT_MARKDOWN = """
# Instance Selection for Training\n\n
The purpose of this analysis is to assist the user in determining 
the optimal periods for training regression models. 
While the algorithm executions are automatic, the conclusions 
need to be validated to ensure their applicability.\n\n
## Instance Selection by Clustering\n\n
The analysis of operating modes resulted in the identification of **{0}** 
distinct operational regions. Regions **{1}** were selected 
as the most relevant. The bar chart is organized in 
ascending order of the error obtained for each region when trying to predict 
the others. The error value is calculated taking into account the 
correlation between predictions and target values, as well as other factors, 
such as the probability of each operating mode.\n\n
![Histogram](data:image/png;base64,{2})
![Clusters](data:image/png;base64,{3})
## Instance Optimization\n\n
After obtaining a new dataset composed only of regions **{1}**,
a second analysis was conducted to identify the window size 
that leads to the lowest prediction errors. This analysis showed
that, for the considered window, the model reaches its maximum predictive 
power at **{6}**. The prediction error for the window resolution 
used (**{7}**) is shown in the following figure. 
![MSE](data:image/png;base64,{4})
![IS_Optimized](data:image/png;base64,{5})
"""


class InstanceSelectionLib:
    """
    Library for instance selection for regression model training.
    
    This class provides methods to identify optimal training periods
    through clustering and sliding window analysis.
    
    Attributes:
        preprocess_list: List of preprocessing steps.
        pipeline_list: List of pipeline steps.
        max_clusters: Maximum number of clusters for k-means.
        min_window: Minimum window size.
        resolution: Resolution for sliding window analysis.
        tolerance: Tolerance for minimum MSE point selection.
        variance_threshold: Explained variance threshold for PCA.
        percentile_limit: Percentile for relevant cluster determination.
        min_size_cluster: Minimum valid cluster size.
        remove_outlier: Flag for outlier removal.
        spike_threshold: Threshold for change detection in MSE profile.
        smartmonitor_output: Dictionary where generated figures are stored.
        show_figures: Flag for figure display.
    """

    def __init__(
        self,
        preprocess_list: Optional[List[str]] = None,
        pipeline_list: Optional[List[str]] = None,
        max_clusters: int = DEFAULT_MAX_CLUSTERS,
        min_window: int = DEFAULT_MIN_WINDOW,
        resolution: int = DEFAULT_RESOLUTION,
        tolerance: float = DEFAULT_TOLERANCE,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        percentile_limit: float = DEFAULT_PERCENTILE_LIMIT,
        min_size_cluster: int = DEFAULT_MIN_SIZE_CLUSTER,
        remove_outlier: bool = True,
        spike_threshold: int = DEFAULT_SPIKE_THRESHOLD,
        smartmonitor_output: Optional[Dict] = None,
        show_figures: bool = True
    ):
        """
        Initialize the instance selection library.
        
        Args:
            preprocess_list: List of preprocessing steps.
                Valid options: 'transform', 'remove_zero_std', 'scaled'
            pipeline_list: List of pipeline steps.
                Valid options: 'preprocess_data', 'kmeans_optimize_clusters',
                'cluster_regressor', 'plot_results', 'sliding_window_regression',
                'plot_sliding_window_results'
            max_clusters: Maximum number of clusters for k-means.
            min_window: Minimum window size.
            resolution: Resolution for sliding window analysis.
            tolerance: Tolerance for minimum MSE point selection.
            variance_threshold: Explained variance threshold for PCA.
            percentile_limit: Percentile for relevant cluster determination.
            min_size_cluster: Minimum valid cluster size.
            remove_outlier: Flag for outlier removal.
            spike_threshold: Threshold for change detection in MSE profile.
            smartmonitor_output: Dictionary where generated figures are stored.
            show_figures: Flag for figure display.
        """
        # Parameter validation
        self._validate_init_parameters(
            max_clusters, min_window, resolution, tolerance,
            variance_threshold, percentile_limit, min_size_cluster, spike_threshold
        )
        
        self.preprocess_list = preprocess_list or ['transform', 'remove_zero_std']
        self.pipeline_list = pipeline_list or [
            'preprocess_data', 'kmeans_optimize_clusters',
            'cluster_regressor', 'plot_results',
            'sliding_window_regression', 'plot_sliding_window_results'
        ]
        self.max_clusters = max_clusters
        self.min_window = min_window
        self.resolution = resolution
        self.tolerance = tolerance
        self.variance_threshold = variance_threshold
        self.percentile_limit = percentile_limit
        self.min_size_cluster = min_size_cluster
        self.remove_outlier = remove_outlier
        self.spike_threshold = spike_threshold
        self.smartmonitor_output = smartmonitor_output or {}
        self.show_figures = show_figures

    def _validate_init_parameters(
        self,
        max_clusters: int,
        min_window: int,
        resolution: int,
        tolerance: float,
        variance_threshold: float,
        percentile_limit: float,
        min_size_cluster: int,
        spike_threshold: int
    ) -> None:
        """Validate initialization parameters."""
        if max_clusters < 2:
            raise ValueError("max_clusters must be greater than 1")
        if min_window <= 0:
            raise ValueError("min_window must be greater than 0")
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1")
        if not 0 <= tolerance <= 1:
            raise ValueError("tolerance must be between 0 and 1")
        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")
        if not 0 < percentile_limit <= 100:
            raise ValueError("percentile_limit must be between 0 and 100")
        if min_size_cluster < 1:
            raise ValueError("min_size_cluster must be greater than 0")
        if spike_threshold < 0:
            raise ValueError("spike_threshold must be greater than or equal to 0")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame data.

        Args:
            df: DataFrame to preprocess.

        Returns:
            DataFrame with preprocessed data.
            
        Raises:
            ValueError: If the DataFrame is empty.
            TypeError: If the input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        try:
            # Copy to avoid modifying the original
            df_processed = df.copy()
            
            # Convert index to datetime if possible
            df_processed.index = pd.to_datetime(df_processed.index)

            if 'transform' in self.preprocess_list:
                # Convert to numeric and fill missing values
                df_processed = df_processed.apply(pd.to_numeric, errors='coerce')
                df_processed = df_processed.ffill().bfill()
    
            if 'remove_zero_std' in self.preprocess_list:
                # Remove columns with zero standard deviation
                std_values = df_processed.std()
                zero_std_cols = std_values[std_values == 0].index.tolist()
                if zero_std_cols:
                    print(f"Removing {len(zero_std_cols)} columns with zero standard deviation")
                    df_processed = df_processed.drop(zero_std_cols, axis=1)
            
            return df_processed.sort_index()
            
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {str(e)}") from e

    def kmeans_optimize_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize the number of clusters in K-Means using the elbow method.
    
        Args:
            df: Data to be clustered.
    
        Returns:
            DataFrame with an additional 'Clusters' column containing cluster labels.
    
        Raises:
            ValueError: If parameters are invalid or data is insufficient.
            TypeError: If the input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if self.max_clusters < 2:
            raise ValueError('Maximum number of clusters must be greater than 1')
        
        if len(df) < self.max_clusters:
            raise ValueError(f"Insufficient data for {self.max_clusters} clusters")
            
        try:
            # Prepare data
            df_processed = df.copy()
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_processed)
            df_scaled = pd.DataFrame(
                df_scaled, 
                columns=df_processed.columns, 
                index=df_processed.index
            )
            
            # Compute initial inertia for normalization
            initial_kmeans = KMeans(
                n_clusters=self.max_clusters, 
                init='k-means++', 
                random_state=42, 
                n_init=10
            )
            initial_inertia = initial_kmeans.fit(df_scaled).inertia_
            normalization_factor = int(math.log10(initial_inertia))
            
            # Find optimal number of clusters
            objective_values = []
            optimal_clusters = self._find_optimal_clusters(
                df_scaled, normalization_factor, objective_values
            )
            
            # Apply k-means with optimal number
            kmeans_optimal = KMeans(
                n_clusters=optimal_clusters, 
                init='k-means++', 
                random_state=42, 
                n_init=10
            )
            df_processed['Clusters'] = kmeans_optimal.fit_predict(df_scaled)
            
            return df_processed
            
        except (ValueError, TypeError) as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error during cluster optimization: {str(e)}") from e

    def _find_optimal_clusters(
        self, 
        df_scaled: pd.DataFrame, 
        normalization_factor: int, 
        objective_values: List[float]
    ) -> int:
        """
        Find the optimal number of clusters using the elbow method.
        
        Args:
            df_scaled: Scaled data.
            normalization_factor: Normalization factor based on initial inertia.
            objective_values: List to store objective function values.
            
        Returns:
            Optimal number of clusters.
        """
        for k in range(2, self.max_clusters):
            kmeans = KMeans(
                n_clusters=k, 
                init='k-means++', 
                random_state=42, 
                n_init=10
            )
            kmeans.fit(df_scaled)
            current_objective = kmeans.inertia_ + (k * (10 ** normalization_factor))
            objective_values.append(current_objective)
            
            # Stopping criterion: if the objective value increases
            if len(objective_values) > 1 and current_objective > objective_values[-2]:
                break
        
        return np.argmin(objective_values) + 2

    def cluster_regressor(
        self, 
        df: pd.DataFrame, 
        target: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """
        Perform per-cluster regression and return clusters sorted by score.

        Args:
            df: DataFrame containing the data.
            target: Name of the target variable in the DataFrame, or None
                to use PCA.

        Returns:
            List of tuples ``(cluster_id, score)`` sorted by score
            in ascending order.
            
        Raises:
            ValueError: If the target variable is not in the DataFrame columns.
        """
        if target is not None and target not in df.columns:
            raise ValueError(f"Target '{target}' is not in the DataFrame columns")
        
        # Remove outliers if needed
        df_filtered = self._filter_outliers(df) if self.remove_outlier else df.copy()
        
        # Compute scores for each cluster
        cluster_scores = self._calculate_cluster_scores(df_filtered, target)
        
        # Sort and filter valid results
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])
        return [(cluster, score) for cluster, score in sorted_clusters 
                if not math.isnan(score)]

    def _filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove small clusters considered as outliers.

        Clusters with a number of elements less than or equal to
        ``min_size_cluster`` are discarded from the DataFrame.

        Args:
            df: DataFrame containing the 'Clusters' column.

        Returns:
            DataFrame without the outlier clusters.

        Raises:
            KeyError: If the 'Clusters' column does not exist in the DataFrame.
        """
        if 'Clusters' not in df.columns:
            raise KeyError("'Clusters' column not found in the DataFrame")

        cluster_counts = df["Clusters"].value_counts()
        clusters_to_drop = cluster_counts[
            cluster_counts <= self.min_size_cluster
        ].index.tolist()
        
        if clusters_to_drop:
            print(f"Removing {len(clusters_to_drop)} outlier clusters")
            
        return df[~df["Clusters"].isin(clusters_to_drop)]

    def _calculate_cluster_scores(
        self, 
        df: pd.DataFrame, 
        target: Optional[str]
    ) -> Dict[int, float]:
        """
        Compute scores for each cluster based on regression or PCA.
        
        Args:
            df: Filtered DataFrame.
            target: Name of the target variable, or None for PCA.
            
        Returns:
            Dictionary mapping cluster IDs to their scores.
        """
        cluster_scores = {}
        frequencies = df["Clusters"].value_counts(normalize=True)
        
        for cluster in df['Clusters'].unique():
            try:
                score = self._evaluate_cluster(df, cluster, target, frequencies)
                cluster_scores[cluster] = score
            except Exception as e:
                print(f"Error evaluating cluster {cluster}: {e}")
                continue
                
        return cluster_scores

    def _evaluate_cluster(
        self, 
        df: pd.DataFrame, 
        cluster: int, 
        target: Optional[str], 
        frequencies: pd.Series
    ) -> float:
        """
        Evaluate a specific cluster using regression or PCA.
        
        Args:
            df: Complete DataFrame.
            cluster: ID of the cluster to evaluate.
            target: Name of the target variable, or None.
            frequencies: Cluster frequencies.
            
        Returns:
            Cluster score.
        """
        # Prepare cluster data
        cluster_df = df[df['Clusters'] == cluster].copy()
        test_df = df[df['Clusters'] != cluster].copy()
        
        # Scale data
        feature_cols = [col for col in cluster_df.columns if col != 'Clusters']
        scaler = StandardScaler()
        cluster_df[feature_cols] = scaler.fit_transform(cluster_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # Compute logarithmic frequency
        log_freq = abs(math.log(frequencies[cluster])) if frequencies[cluster] > 0 else 0
        
        if target is None:
            return self._evaluate_cluster_pca(cluster_df, test_df, log_freq)
        else:
            return self._evaluate_cluster_regression(
                cluster_df, test_df, target, log_freq
            )

    def _evaluate_cluster_pca(
        self, 
        cluster_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        log_freq: float
    ) -> float:
        """Evaluate cluster quality using PCA reconstruction error.

        Fits a PCA model on the cluster data and measures the mean squared
        error when reconstructing the test data (remaining clusters),
        weighted by the cluster's logarithmic frequency.

        Args:
            cluster_df: Cluster data (already scaled), containing the
                'Clusters' column.
            test_df: Remaining clusters data (already scaled), containing
                the 'Clusters' column.
            log_freq: Logarithmic weight derived from cluster frequency.

        Returns:
            Weighted reconstruction mean squared error.
        """
        # Configure PCA
        pca = PCA().fit(cluster_df.drop('Clusters', axis=1))
        
        if self.variance_threshold is not None:
            explained_variance = pca.explained_variance_ratio_
            cumsum_var = np.cumsum(explained_variance)
            indices = np.where(cumsum_var >= self.variance_threshold)[0]
            num_components = (
                indices[0] + 1 if len(indices) > 0 
                else len(explained_variance)
            )
            pca = PCA(n_components=num_components).fit(
                cluster_df.drop('Clusters', axis=1)
            )
        
        # Compute reconstruction error
        X_test = test_df.drop('Clusters', axis=1)
        X_reconstructed = pca.inverse_transform(pca.transform(X_test))
        mse = mean_squared_error(X_test, X_reconstructed) * log_freq
        
        return mse

    def _evaluate_cluster_regression(
        self, 
        cluster_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        target: str, 
        log_freq: float
    ) -> float:
        """Evaluate cluster quality using supervised regression.

        Trains a ``DecisionTreeRegressor`` on the cluster data and predicts
        the target variable on the remaining clusters. The score combines
        MSE, Pearson correlation, and the cluster's logarithmic frequency.

        Args:
            cluster_df: Cluster data (already scaled), containing the
                'Clusters' and ``target`` columns.
            test_df: Remaining clusters data (already scaled), containing
                the 'Clusters' and ``target`` columns.
            target: Name of the target column for regression.
            log_freq: Logarithmic weight derived from cluster frequency.

        Returns:
            Weighted score: ``MSE * log_freq / |pearson_r|``.
        """
        # Prepare data for regression
        X_cluster = cluster_df.drop([target, 'Clusters'], axis=1)
        y_cluster = cluster_df[target]
        
        X_test = test_df.drop([target, 'Clusters'], axis=1)
        y_test = test_df[target]
        
        # Train regressor
        regressor = DecisionTreeRegressor(random_state=42)
        regressor.fit(X_cluster, y_cluster)
        
        # Make predictions and compute metrics
        y_pred = regressor.predict(X_test)
        pearson_r, _ = pearsonr(y_test, y_pred)
        
        # Compute final score
        mse = mean_squared_error(y_test, y_pred)
        score = mse * log_freq / max(abs(pearson_r), MIN_FREQUENCY_THRESHOLD)
        
        return score

    def detect_derivative_spikes(
        self, 
        signal: Union[List[float], np.ndarray], 
        threshold_std: float
    ) -> np.ndarray:
        """
        Detect negative spikes in the derivative (abrupt drops) using percentile.

        Args:
            signal: One-dimensional time series.
            threshold_std: Percentile to define an abrupt drop.

        Returns:
            Indices of the points where abrupt drops occurred.
            
        Raises:
            ValueError: If the signal is too short or threshold is invalid.
        """
        if len(signal) < 2:
            raise ValueError("Signal must have at least 2 points")
        
        if not 0 <= threshold_std <= 100:
            raise ValueError("threshold_std must be between 0 and 100")
        
        signal_array = np.asarray(signal)
        derivative = np.diff(signal_array)
    
        threshold_value = np.percentile(derivative, threshold_std)
        spike_indices = np.where(derivative < threshold_value)[0] + 1
    
        return spike_indices

    def sliding_window_regression(
        self, 
        df: pd.DataFrame, 
        target: Optional[str] = None
    ) -> Tuple[List[float], int, np.ndarray]:
        """Perform sliding window regression analysis.

        For each window size (from ``min_window`` up to ``n_rows``, with
        step determined by ``resolution``), trains a model on the initial
        data and computes the prediction error on the remaining data. When
        ``target`` is ``None``, uses PCA reconstruction error; otherwise
        uses ``DecisionTreeRegressor``.

        Args:
            df: Input DataFrame (without the 'Clusters' column).
            target: Name of the target column. If ``None``, uses PCA with
                ``self.variance_threshold`` for unsupervised evaluation.

        Returns:
            Tuple containing:
            - test_errors: List of test errors for each window.
            - min_error_idx: Index (in DataFrame rows) of the minimum error.
            - spike_idxs: NumPy array with indices of abrupt drops in the
              error profile.

        Raises:
            ValueError: If ``target`` does not exist in the columns, if
                ``min_window <= 0``, if the DataFrame has fewer rows
                than ``min_window``, or if ``resolution <= 1``.
        """
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' was not found in "
                             "the dataframe")
    
        if self.min_window <= 0:
            raise ValueError("Minimum window size must be greater than zero.")
    
        if df.shape[0] < self.min_window:
            raise ValueError(f"The dataframe has {df.shape[0]} rows, which is "
                             f"less than the minimum window size of "
                             f"{self.min_window}.")
    
        if self.resolution <= 1:
            raise ValueError("Resolution must be greater than one.")
    
        test_errors = []
        n_rows = df.shape[0]
        interval = max(1, int(n_rows / self.resolution))
        min_error = float('inf')
        min_error_idx = -1
        scaler = StandardScaler()
    
        for i in range(self.min_window, n_rows, interval):
            train_data = scaler.fit_transform(df.iloc[:i])
            train_data = pd.DataFrame(train_data,
                                     columns=df.columns,
                                     index=df.iloc[:i].index)
            test_data = scaler.transform(df.iloc[i:])
            test_data = pd.DataFrame(test_data,
                                     columns=df.columns,
                                     index=df.iloc[i:].index)
            if target is None and self.variance_threshold is None:
                raise ValueError("Either the target variable or the explained "
                                 "variance must be provided.")
    
            if target is None:
                pca = PCA().fit(train_data.dropna())
                explained_variance_ratio = pca.explained_variance_ratio_
                cumsum_var = np.cumsum(explained_variance_ratio)
                indices = np.where(cumsum_var >= self.variance_threshold)[0]
                num_components = (
                    indices[0] + 1 if len(indices) > 0 
                    else len(explained_variance_ratio)
                )
                pca = PCA(n_components=num_components).fit(train_data.dropna())
                reg_tree = None
            else:
                reg_tree = DecisionTreeRegressor(max_features=0.3, random_state=42)
                reg_tree.fit(train_data.drop(target, axis=1), train_data[target])
            if reg_tree is not None:
                preds = reg_tree.predict(test_data.drop(target, axis=1))
                test_error = mean_squared_error(test_data[target], preds)
            else:
                preds = pca.inverse_transform(pca.transform(test_data))
                test_error = mean_squared_error(test_data, preds)
            test_errors.append(test_error)
            if test_error < min_error:
                min_error = test_error
                min_error_idx = i
        spike_idxs = self.detect_derivative_spikes(test_errors, threshold_std=self.spike_threshold)    
        return test_errors, min_error_idx, spike_idxs

    def plot_results(
        self, 
        df: pd.DataFrame, 
        sorted_clusters: List[Tuple[int, float]]
    ) -> None:
        """Plot the cluster regression analysis results.

        Generates two plots:
        1. Weighted error histogram per cluster (logarithmic scale),
           with a percentile line.
        2. Time series of each variable colored by cluster.

        The plots are stored in ``self.smartmonitor_output`` under the
        keys ``'ranking_clusters'`` and ``'trends_with_clusters'``.

        Args:
            df: DataFrame with the data and the 'Clusters' column.
            sorted_clusters: List of tuples ``(cluster_id, weighted_error)``
                sorted by ascending error.

        Raises:
            ValueError: If ``sorted_clusters`` is empty or
                ``percentile_limit <= 0``.
        """
        if not sorted_clusters:
            raise ValueError("The sorted clusters list is empty.")
            
        if self.percentile_limit <= 0:
            raise ValueError("Percentile must be greater than zero.")
        
        x = [f'Cluster {str(cluster[0])}' for cluster in sorted_clusters]
        y = [cluster[1] for cluster in sorted_clusters]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(x, y, width=1, edgecolor="white", 
                linewidth=0.7, color='xkcd:sky blue')
        percentile_lim = np.percentile(np.array(sorted_clusters)[:, 1], 
                                       self.percentile_limit)
        ax.axhline(y=percentile_lim, linestyle='--', color='b')
        ax.text(x[0], percentile_lim*1.25, 
                 f'Value corresponding to percentile {self.percentile_limit}', 
                 fontsize = 14)
        ax.set_ylabel('Weighted error')
        ax.set_xticks(x)
        ax.set_xticklabels(labels=x, rotation=90, ha='right')
        ax.set_yscale('log')
        if self.show_figures:
            plt.show()
        self.smartmonitor_output["ranking_clusters"] = fig 
        
        cluster_col = 'Clusters'
        clusters = df[cluster_col].unique()
        fig, axs = plt.subplots(nrows=df.shape[1]-1, figsize=(10, 3*df.shape[1]))
        for i, col in enumerate(df.columns):
            if col != cluster_col:
                ax = axs[i]
                for cluster in clusters:
                    subset = df[df[cluster_col] == cluster]
                    ax.plot(subset.index, subset[col], label=f'Cluster {cluster}', 
                            linestyle='none', marker='.')
                    date_fmt = mdates.DateFormatter('%Y-%m-%d')
                    ax.xaxis.set_major_formatter(date_fmt)
                    ax.set_ylabel(col)
                    fig.autofmt_xdate()
                    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        if self.show_figures:
            plt.show()
        self.smartmonitor_output["trends_with_clusters"] = fig
    
    def plot_sliding_window_results(
        self, 
        df: pd.DataFrame, 
        test_errors: List[float]
    ) -> None:
        """Plot the sliding window analysis results.

        Generates two plots:
        1. MSE curve per window size, highlighting the minimum error
           point (considering ``self.tolerance``).
        2. Time series of each variable with a vertical line indicating
           the optimal cutoff point.

        The plots are stored in ``self.smartmonitor_output`` under the
        keys ``'minimum_error'`` and ``'minimum_error_tags'``.

        Args:
            df: DataFrame with time series (without 'Clusters' column).
            test_errors: List of MSE errors returned by
                ``sliding_window_regression``.

        Raises:
            TypeError: If ``test_errors`` is not a list, ``df`` is not a
                DataFrame, or ``self.tolerance`` is not numeric.
            ValueError: If ``test_errors`` has fewer than 2 elements or
                ``df`` is empty.
        """
        if not isinstance(test_errors, list):
            raise TypeError("Input must be a list.")
        
        if len(test_errors) < 2:
            raise ValueError("Input data must contain at least two elements.")
        
        if not isinstance(self.tolerance, (float, int)):
            raise TypeError("self.tolerance must be numeric")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a DataFrame.")

        if df.empty or df.shape[0] < 2:
            raise ValueError("df must contain at least two rows of data.")
        
        for j in range(len(test_errors)):
            if test_errors[j] <= min(test_errors)*(1+self.tolerance):
                minsize_point = j
                minsize = test_errors[minsize_point]
                break    
        y = min(
            int(df.shape[0] / self.resolution) * minsize_point, 
            len(df) - 1
        )
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(test_errors, color='xkcd:sky blue', linewidth=2.0)
        ax.set_ylabel('MSE')
        ax.axhline(y=minsize, color='k', linestyle='--')
        ax.plot(minsize_point, minsize, 'ro', ms=12, mec='black')
        if self.show_figures:
            plt.show()
        self.smartmonitor_output["minimum_error"] = fig
        
        fig, axs = plt.subplots(nrows=df.shape[1], figsize=(10, 3*df.shape[1]))
        
        for i, col in enumerate(df.columns):
            axs[i].plot(df.index, df[col], linestyle='none', 
                        marker='.', color='xkcd:sky blue')
            axs[i].axvline(x=df.index[y], color='red', 
                           linestyle='--', linewidth=2)
            axs[i].set_ylabel(col)
            fig.autofmt_xdate()
            axs[i].xaxis.set_major_locator(plt.MaxNLocator(10))
        
        if self.show_figures:
            plt.show()
        self.smartmonitor_output["minimum_error_tags"] = fig
    
    def list_dates_from_clusters(
        self, 
        clusters_to_keep: List[int], 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract consecutive date intervals for specific clusters.

        Iterates through the DataFrame sorted by index and groups
        consecutive rows belonging to the desired clusters, recording the
        start and end dates of each continuous segment. Single-row
        segments are discarded.

        Args:
            clusters_to_keep: List of cluster IDs to consider.
            df: DataFrame containing the 'Clusters' column, with a
                temporal index.

        Returns:
            DataFrame with columns ``'Initial date'``, ``'Final date'``,
            and ``'Cluster'``.

        Raises:
            TypeError: If ``clusters_to_keep`` is not a list or ``df`` is
                not a DataFrame.
            KeyError: If the 'Clusters' column does not exist in the DataFrame.
        """
        if not isinstance(clusters_to_keep, list):
            raise TypeError("clusters_to_keep must be a list")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if 'Clusters' not in df.columns:
            raise KeyError("'Clusters' column not found in the DataFrame")

        result_rows = []
        current_cluster = None
        start_index = None
        last_index = None
        
        for index, row in df.iterrows():
            if row['Clusters'] in clusters_to_keep:
                if current_cluster is None:
                    current_cluster = row['Clusters']
                    start_index = index
                elif current_cluster == row['Clusters']:
                    last_index = index
                else:
                    if last_index is not None:
                        result_rows.append({'Initial date': start_index,
                                            'Final date': last_index,
                                            'Cluster': current_cluster})
                    current_cluster = row['Clusters']
                    start_index = index
                    last_index = index
            else:
                if current_cluster is not None and last_index is not None:
                    result_rows.append({'Initial date': start_index,
                                        'Final date': last_index,
                                        'Cluster': current_cluster})
                current_cluster = None
                start_index = None
                last_index = None

        if current_cluster is not None and last_index is not None:
            result_rows.append({'Initial date': start_index,
                                'Final date': last_index,
                                'Cluster': current_cluster})

        result_df = pd.DataFrame(result_rows)
        if result_df.empty:
            return result_df
        result_df = result_df[result_df["Initial date"] != result_df["Final date"]]
        return result_df

    def results_markdown(
        self, 
        sorted_clusters: List[Tuple[int, float]], 
        clusters_to_keep: List[int], 
        data_final
    ) -> str:
        """Generate a complete Markdown report with embedded images.

        Encodes the plots stored in ``self.smartmonitor_output`` to Base64
        and inserts them into the ``RESULT_MARKDOWN`` template.

        Args:
            sorted_clusters: List of tuples ``(cluster_id, error)``.
            clusters_to_keep: IDs of the selected clusters.
            data_final: Date/index corresponding to the optimal sliding
                window cutoff point.

        Returns:
            Markdown string with report and embedded images.

        Raises:
            KeyError: If any expected image is not in
                ``self.smartmonitor_output``.
        """
        image_names = [
        "ranking_clusters",
        "trends_with_clusters",
        "minimum_error",
        "minimum_error_tags"
        ]
        encoded_images = []

        for image_name in image_names:
            fig = self.smartmonitor_output[image_name]
            buffer_image = io.BytesIO()
            fig.savefig(buffer_image, dpi=150, format="png", bbox_inches="tight")
            buffer_image.seek(0)
            encoded_image = base64.b64encode(buffer_image.read()).decode()
            encoded_images.append(encoded_image)

        return RESULT_MARKDOWN.format(
            len(sorted_clusters),
            clusters_to_keep,
            *encoded_images,
            data_final,
            self.resolution
        )

    def cluster_analysis(
        self, 
        df: pd.DataFrame, 
        target: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[List], Optional[pd.DataFrame], Optional[str]]:
        """
        Perform cluster analysis only.
        
        Args:
            df: DataFrame to be analyzed.
            target: Name of the target column. If None, it will not be used.
        
        Returns:
            Tuple containing:
            - df_processed: Preprocessed DataFrame with clusters.
            - sorted_clusters: List of clusters sorted by error value.
            - df_results: DataFrame with start and end dates of cluster windows.
            - results_markdown: String containing the Markdown report.
        """
        try:
            # Initialize variables
            sorted_clusters = None
            clusters_to_keep = []
            df_results = None
            results_markdown = None
            
            # Data preprocessing
            df_processed = self.preprocess_data(df)
            
            # Cluster optimization
            df_processed = self.kmeans_optimize_clusters(df_processed)
            
            # Per-cluster regression
            sorted_clusters = self.cluster_regressor(df_processed, target)
            if sorted_clusters:
                percentile_lim = np.percentile(
                    np.array(sorted_clusters)[:, 1], 
                    self.percentile_limit
                )
                clusters_to_keep = [
                    sorted_clusters[x][0] 
                    for x in range(len(sorted_clusters)) 
                    if sorted_clusters[x][1] <= percentile_lim
                ]
                df_results = self.list_dates_from_clusters(clusters_to_keep, df_processed)
            
            # Plot cluster results
            if sorted_clusters is not None:
                self.plot_results(df_processed, sorted_clusters)
            else:
                print("No cluster analysis was performed.")
            
            # Generate cluster analysis markdown report
            if sorted_clusters is not None and hasattr(self, 'smartmonitor_output'):
                try:
                    results_markdown = self._generate_cluster_markdown(sorted_clusters, clusters_to_keep)
                except Exception as e:
                    print(f"Error generating cluster markdown report: {e}")
                    results_markdown = None

            return df_processed, sorted_clusters, df_results, results_markdown
    
        except Exception as e:
            print(f"Error during cluster analysis: {str(e)}")
            return df, None, None, None

    def window_analysis(
        self, 
        df: pd.DataFrame, 
        target: Optional[str] = None
    ) -> Tuple[Optional[List], Optional[int], Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        """
        Perform sliding window analysis only.
        
        Args:
            df: DataFrame to be analyzed (already preprocessed).
            target: Name of the target column. If None, it will not be used.
        
        Returns:
            Tuple containing:
            - test_errors: List of errors from the sliding window analysis.
            - min_error_idx: Index of the minimum error.
            - spike_idxs: NumPy array with indices where changes were detected.
            - df_optimized: DataFrame filtered up to data_final (optimized dataset).
            - results_markdown: String containing the Markdown report.
        """
        try:
            # Initialize variables
            test_errors = None
            min_error_idx = None
            spike_idxs = None
            df_optimized = None
            results_markdown = None
            
            # Prepare data (remove Clusters column if present)
            df_for_sliding = df.copy()
            if 'Clusters' in df_for_sliding.columns:
                df_for_sliding = df_for_sliding.drop('Clusters', axis=1)
            
            # Sliding window regression
            test_errors, min_error_idx, spike_idxs = self.sliding_window_regression(
                df_for_sliding, target
            )
            
            # Compute data_final and create optimized dataset
            if test_errors is not None:
                minsize_point = 0
                for j in range(len(test_errors)):
                    if test_errors[j] <= min(test_errors) * (1 + self.tolerance):
                        minsize_point = j
                        break    
                
                y = int(df_for_sliding.shape[0] / self.resolution) * minsize_point
                if y < len(df_for_sliding):
                    data_final = df_for_sliding.index[y]
                    # Create optimized dataset filtered up to data_final
                    df_optimized = df_for_sliding.loc[df_for_sliding.index <= data_final].copy()
            
            # Plot sliding window results
            if test_errors is not None:
                self.plot_sliding_window_results(df_for_sliding, test_errors)
            else:
                print("No regression was performed.")
            
            # Generate window analysis markdown report
            if test_errors is not None and hasattr(self, 'smartmonitor_output'):
                try:
                    # Find minimum error point for the report
                    minsize_point = 0
                    for j in range(len(test_errors)):
                        if test_errors[j] <= min(test_errors) * (1 + self.tolerance):
                            minsize_point = j
                            break    
                    
                    y = int(df_for_sliding.shape[0] / self.resolution) * minsize_point
                    if y < len(df_for_sliding):
                        data_final = df_for_sliding.index[y]
                        results_markdown = self._generate_window_markdown(test_errors, data_final)
                except Exception as e:
                    print(f"Error generating window markdown report: {e}")
                    results_markdown = None

            return test_errors, min_error_idx, spike_idxs, df_optimized, results_markdown
    
        except Exception as e:
            print(f"Error during window analysis: {str(e)}")
            return None, None, np.array([]), None, None

    def _generate_cluster_markdown(self, sorted_clusters: List, clusters_to_keep: List) -> str:
        """
        Generate a markdown report specific to cluster analysis.
        
        Args:
            sorted_clusters: Sorted list of clusters.
            clusters_to_keep: List of selected clusters.
            
        Returns:
            Markdown report string.
        """
        cluster_markdown = f"""
# Clustering Analysis

## Instance Selection by Clustering

The analysis of operating modes resulted in the identification of **{len(sorted_clusters)}** 
distinct operational regions. Regions **{clusters_to_keep}** were selected 
as the most relevant. The bar chart is organized in 
ascending order of the error obtained for each region when trying to predict 
the others. The error value is calculated taking into account the 
correlation between predictions and target values, as well as other factors, 
such as the probability of each operating mode.

"""
        
        # Add images if available
        if "ranking_clusters" in self.smartmonitor_output:
            fig = self.smartmonitor_output["ranking_clusters"]
            buffer_image = io.BytesIO()
            fig.savefig(buffer_image, dpi=150, format="png", bbox_inches="tight")
            buffer_image.seek(0)
            encoded_image = base64.b64encode(buffer_image.read()).decode()
            cluster_markdown += f"![Histogram](data:image/png;base64,{encoded_image})\n\n"
        
        if "trends_with_clusters" in self.smartmonitor_output:
            fig = self.smartmonitor_output["trends_with_clusters"]
            buffer_image = io.BytesIO()
            fig.savefig(buffer_image, dpi=150, format="png", bbox_inches="tight")
            buffer_image.seek(0)
            encoded_image = base64.b64encode(buffer_image.read()).decode()
            cluster_markdown += f"![Clusters](data:image/png;base64,{encoded_image})\n\n"
        
        return cluster_markdown

    def _generate_window_markdown(self, test_errors: List, data_final) -> str:
        """
        Generate a markdown report specific to sliding window analysis.
        
        Args:
            test_errors: List of test errors.
            data_final: Optimized final date.
            
        Returns:
            Markdown report string.
        """
        window_markdown = f"""
# Sliding Window Analysis

## Instance Optimization

An analysis was conducted to identify the window size 
that leads to the lowest prediction errors. This analysis showed
that, for the considered window, the model reaches its maximum predictive 
power at **{data_final}**. The prediction error for the window resolution 
used (**{self.resolution}**) is shown in the following figure.

"""
        
        # Add images if available
        if "minimum_error" in self.smartmonitor_output:
            fig = self.smartmonitor_output["minimum_error"]
            buffer_image = io.BytesIO()
            fig.savefig(buffer_image, dpi=150, format="png", bbox_inches="tight")
            buffer_image.seek(0)
            encoded_image = base64.b64encode(buffer_image.read()).decode()
            window_markdown += f"![MSE](data:image/png;base64,{encoded_image})\n\n"
        
        if "minimum_error_tags" in self.smartmonitor_output:
            fig = self.smartmonitor_output["minimum_error_tags"]
            buffer_image = io.BytesIO()
            fig.savefig(buffer_image, dpi=150, format="png", bbox_inches="tight")
            buffer_image.seek(0)
            encoded_image = base64.b64encode(buffer_image.read()).decode()
            window_markdown += f"![IS_Optimized](data:image/png;base64,{encoded_image})\n\n"
        
        return window_markdown
    
    def full_analysis(
        self, 
        df: pd.DataFrame, 
        target: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[List], Optional[List], Optional[int], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """
        Perform a complete data analysis on a DataFrame using the provided
        pipeline list.
        
        Args:
            df: DataFrame to be analyzed.
            target: Name of the target column. If None, it will not be used
                in any pipeline step.
        
        Returns:
            Tuple containing:
            - df: Preprocessed DataFrame.
            - sorted_clusters: List of clusters sorted by error value.
            - test_errors: Errors obtained by the sliding window regressor.
            - min_error_idx: Index corresponding to the minimum error.
            - df_results: DataFrame with start and end dates of windows.
            - df_optimized: DataFrame filtered up to data_final (optimized dataset).
            - results_markdown: String containing the Markdown report.
        """
        try:
            # Initialize variables
            sorted_clusters = None
            test_errors = None
            min_error_idx = None
            spike_idxs = None
            df_results = None
            df_optimized = None
            df_reduced = None
            results_markdown = None
            clusters_to_keep = []
            data_final = None
            
            # Data preprocessing
            df_processed = self.preprocess_data(df)
    
            if not self.pipeline_list:
                return df_processed, None, None, None, None, None, None
    
            # Cluster optimization
            if 'kmeans_optimize_clusters' in self.pipeline_list:
                df_processed = self.kmeans_optimize_clusters(df_processed)
    
            # Per-cluster regression
            if 'cluster_regressor' in self.pipeline_list:
                sorted_clusters = self.cluster_regressor(df_processed, target)
                if sorted_clusters:
                    percentile_lim = np.percentile(
                        np.array(sorted_clusters)[:, 1], 
                        self.percentile_limit
                    )
                    clusters_to_keep = [
                        sorted_clusters[x][0] 
                        for x in range(len(sorted_clusters)) 
                        if sorted_clusters[x][1] <= percentile_lim
                    ]
                    df_reduced = df_processed.loc[
                        df_processed['Clusters'].isin(clusters_to_keep)
                    ]
                    df_results = self.list_dates_from_clusters(clusters_to_keep, df_processed)
    
            # Plot cluster results
            if 'plot_results' in self.pipeline_list:
                if sorted_clusters is not None:
                    self.plot_results(df_processed, sorted_clusters)
                else:
                    print("No cluster analysis was performed.")
            
            # Sliding window regression
            if 'sliding_window_regression' in self.pipeline_list:
                df_for_sliding = df_processed.copy()
                if sorted_clusters is not None and df_reduced is not None:
                    df_for_sliding = df_reduced.drop('Clusters', axis=1)
                elif 'Clusters' in df_for_sliding.columns:
                    df_for_sliding = df_for_sliding.drop('Clusters', axis=1)
                    
                test_errors, min_error_idx, spike_idxs = self.sliding_window_regression(
                    df_for_sliding, target
                )
                
                # Find minimum error point
                if test_errors:
                    minsize_point = 0
                    for j in range(len(test_errors)):
                        if test_errors[j] <= min(test_errors) * (1 + self.tolerance):
                            minsize_point = j
                            break    
                    
                    y = int(df_for_sliding.shape[0] / self.resolution) * minsize_point
                    if y < len(df_for_sliding):
                        data_final = df_for_sliding.index[y]
                        # Create optimized dataset filtered up to data_final
                        df_optimized = df_for_sliding.loc[df_for_sliding.index <= data_final].copy()
                        
                        # Adjust df_results if needed
                        if df_results is not None and not df_results.empty:
                            closest_date = df_results['Final date'].iloc[
                                (df_results['Final date'] - data_final).abs().argmin()
                            ]
                            df_results = df_results.loc[
                                df_results['Final date'] < closest_date
                            ]
                    
            # Plot sliding window results
            if 'plot_sliding_window_results' in self.pipeline_list:
                if test_errors is not None:
                    df_for_plot = df_processed.copy()
                    if 'Clusters' in df_for_plot.columns:
                        df_for_plot = df_for_plot.drop('Clusters', axis=1)
                    self.plot_sliding_window_results(df_for_plot, test_errors)
                else:
                    print("No regression was performed.")

            # Generate markdown report if possible
            if (sorted_clusters is not None and clusters_to_keep and 
                data_final is not None and hasattr(self, 'smartmonitor_output')):
                try:
                    results_markdown = self.results_markdown(
                        sorted_clusters, clusters_to_keep, data_final
                    )
                except Exception as e:
                    print(f"Error generating markdown report: {e}")
                    results_markdown = None

            return df_processed, sorted_clusters, test_errors, min_error_idx, df_results, df_optimized, results_markdown
    
        except Exception as e:
            print(f"Error during full analysis: {str(e)}")
            # Return default values instead of None
            return df, None, None, None, None, None, None
