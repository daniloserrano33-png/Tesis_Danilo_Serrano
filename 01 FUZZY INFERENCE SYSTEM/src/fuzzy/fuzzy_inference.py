import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, 
    f1_score, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import re
from collections import Counter


def define_loan_purpose_groups() -> Dict[str, List[str]]:
    """Define semantic grouping for loan purpose categories.

    This function consolidates 14 original loan purpose categories into 5 logical
    groups based on financial use case similarity. The groups are: Debt Consolidation,
    Home Related, Business Venture, Major Purchases, and Personal Expenses.

    Returns:
        Dict[str, List[str]]: Dictionary mapping group names to lists of original
                              category names.
    """
    return {
        'Debt_Consolidation': ['debt_consolidation', 'credit_card'],
        'Home_Related': ['home_improvement', 'house', 'moving'],
        'Business_Venture': ['small_business'],
        'Major_Purchases': ['car', 'major_purchase'],
        'Personal_Expenses': [
            'medical', 'wedding', 'educational', 
            'vacation', 'renewable_energy', 'other'
        ]
    }


def apply_semantic_grouping(
    data_series: pd.Series, 
    group_mapping: Dict[str, List[str]]
) -> pd.Series:
    """Apply category consolidation to reduce dimensionality.

    This function maps original categorical values to their consolidated group labels
    using a reverse lookup dictionary built from the group mapping.

    Args:
        data_series (pd.Series): Original categorical data series.
        group_mapping (Dict[str, List[str]]): Group definitions mapping group labels
                                              to lists of original categories.

    Returns:
        pd.Series: Series with consolidated group labels replacing original categories.
    """
    category_to_group = {}
    for group_label, category_list in group_mapping.items():
        for category in category_list:
            category_to_group[category] = group_label
    
    return data_series.map(category_to_group)


def detect_categorical_type(data_series: pd.Series) -> bool:
    """Identify if variable is categorical based on dtype.

    This function checks whether a pandas Series or array has a categorical data type
    by examining if the dtype is object, category, or string-like.

    Args:
        data_series (pd.Series): Pandas Series or array to analyze.

    Returns:
        bool: True if the data type is categorical (object, category, or string),
              False otherwise.
    """
    if isinstance(data_series, pd.Series):
        return (
            data_series.dtype == object or 
            data_series.dtype.name == 'category'
        )
    return data_series.dtype.kind in ['U', 'S', 'O']


def detect_binary_type(data_series: pd.Series) -> bool:
    """Identify if variable contains only binary values.

    This function determines whether a series contains exclusively the values 0.0 and
    1.0 after removing missing values. The series must have exactly two unique values
    that match the binary set.

    Args:
        data_series (pd.Series): Data series to check for binary property.

    Returns:
        bool: True if the series contains only {0.0, 1.0}, False otherwise.
    """
    clean_data = data_series[pd.notna(data_series)]
    
    if len(clean_data) == 0:
        return False
    
    unique_values = np.unique(clean_data)
    
    if len(unique_values) != 2:
        return False
    
    try:
        value_set = {float(v) for v in unique_values}
        return value_set == {0.0, 1.0}
    except (ValueError, TypeError):
        return False


def classify_variable_type(
    data_series: pd.Series,
    category_mappings: Optional[Dict[str, Dict]] = None,
    variable_name: Optional[str] = None
) -> str:
    """Determine variable type for appropriate fuzzy treatment.

    This function classifies a variable as categorical, binary, or numeric. The function
    first checks if the variable name exists in category_mappings to identify pre-encoded
    categorical features. Otherwise, it uses data type detection to classify the variable.

    Args:
        data_series (pd.Series): Variable data to classify.
        category_mappings (Optional[Dict[str, Dict]]): Mappings in format
                                                       {feature: {category: code}} from
                                                       training to identify encoded
                                                       categorical features.
        variable_name (Optional[str]): Variable name to check against mappings.

    Returns:
        str: Variable type classification: 'categorical', 'binary', or 'numeric'.
    """
    if category_mappings and variable_name and variable_name in category_mappings:
        return 'categorical'
    
    if detect_categorical_type(data_series):
        return 'categorical'
    elif detect_binary_type(data_series):
        return 'binary'
    else:
        return 'numeric'


def generate_fallback_cutpoints(
    threshold_values: List[float],
    data_array: np.ndarray
) -> List[float]:
    """Generate fallback cutpoints when optimization fails.

    This function creates a basic cutpoint set using rule thresholds extended by a
    margin calculated from the data range. The margin is set to 30% of the data span
    to ensure coverage beyond the observed values.

    Args:
        threshold_values (List[float]): Threshold values from decision rules.
        data_array (np.ndarray): Clean numeric data array.

    Returns:
        List[float]: Basic cutpoint set with margin extension.
    """
    unique_thresholds = sorted(set(threshold_values))
    
    if len(data_array) > 0:
        data_span = np.max(data_array) - np.min(data_array)
        margin = data_span * 0.3
        
        return [
            np.min(data_array) - margin,
            *unique_thresholds,
            np.max(data_array) + margin
        ]
    else:
        threshold_span = max(unique_thresholds) - min(unique_thresholds)
        margin = max(threshold_span * 0.3, 1.0)
        
        return [
            min(unique_thresholds) - margin,
            *unique_thresholds,
            max(unique_thresholds) + margin
        ]


def guarantee_complete_coverage(
    initial_cutpoints: List[float],
    rule_thresholds: List[float],
    data_array: np.ndarray
) -> List[float]:
    """Ensure cutpoints cover all rule thresholds and data range.

    This function performs three stages of coverage verification: ensuring each threshold
    has cutpoints both below and above, extending coverage to data boundaries, and
    verifying threshold extremes. Additional cutpoints are inserted with appropriate
    margins as needed.

    Args:
        initial_cutpoints (List[float]): Cutpoints from optimization.
        rule_thresholds (List[float]): Thresholds used in decision rules.
        data_array (np.ndarray): Clean numeric data array.

    Returns:
        List[float]: Extended cutpoints with guaranteed coverage of all thresholds
                     and data range.
    """
    if len(initial_cutpoints) == 0:
        data_min, data_max = np.min(data_array), np.max(data_array)
        data_span = data_max - data_min
        margin = data_span * 0.2
        return [data_min - margin, data_max + margin]
    
    cutpoints = sorted(set(initial_cutpoints))
    data_span = np.max(data_array) - np.min(data_array)
    
    for threshold in rule_thresholds:
        has_lower = any(cp < threshold for cp in cutpoints)
        has_upper = any(cp > threshold for cp in cutpoints)
        
        local_gap = data_span * 0.05
        
        if not has_lower:
            cutpoints.append(threshold - local_gap)
        if not has_upper:
            cutpoints.append(threshold + local_gap)
    
    cutpoints = sorted(set(cutpoints))
    
    data_min, data_max = np.min(data_array), np.max(data_array)
    cutpoint_min, cutpoint_max = min(cutpoints), max(cutpoints)
    
    boundary_margin = data_span * 0.15
    
    if cutpoint_min > data_min:
        cutpoints.insert(0, data_min - boundary_margin)
    
    if cutpoint_max < data_max:
        cutpoints.append(data_max + boundary_margin)
    
    if rule_thresholds:
        threshold_min = min(rule_thresholds)
        threshold_max = max(rule_thresholds)
        
        cutpoint_min, cutpoint_max = min(cutpoints), max(cutpoints)
        
        if cutpoint_min > threshold_min:
            cutpoints.insert(0, threshold_min - boundary_margin)
        
        if cutpoint_max < threshold_max:
            cutpoints.append(threshold_max + boundary_margin)
    
    return sorted(list(set(cutpoints)))


def optimize_cutpoints_via_density_clustering(
    rule_thresholds: List[float],
    data_distribution: np.ndarray,
    max_cluster_count: Optional[int] = None
) -> List[float]:
    """Generate cutpoints using density-weighted K-means clustering.

    This function estimates data density using kernel density estimation, extracts
    high-density regions above the 70th percentile, combines them with rule thresholds
    weighted twice, determines optimal cluster count via silhouette score analysis,
    and uses cluster boundaries as cutpoints.

    Args:
        rule_thresholds (List[float]): Threshold values from decision rules.
        data_distribution (np.ndarray): Clean numeric data array.
        max_cluster_count (Optional[int]): Upper limit on clusters, automatically
                                           determined if None.

    Returns:
        List[float]: Optimized cutpoint list from clustering analysis.
    """
    clean_data = data_distribution[~np.isnan(data_distribution)]
    
    if len(clean_data) < 10:
        return generate_fallback_cutpoints(rule_thresholds, clean_data)
    
    density_estimator = gaussian_kde(clean_data, bw_method='scott')
    
    data_min, data_max = np.min(clean_data), np.max(clean_data)
    sample_points = np.linspace(data_min, data_max, 100)
    density_values = density_estimator(sample_points)
    
    density_threshold = np.percentile(density_values, 70)
    high_density_mask = density_values >= density_threshold
    high_density_points = sample_points[high_density_mask]
    
    unique_thresholds = sorted(set(rule_thresholds))
    clustering_points = np.array(
        list(unique_thresholds) * 2 + list(high_density_points)
    ).reshape(-1, 1)
    
    if max_cluster_count is None:
        n_thresholds = len(unique_thresholds)
        min_clusters = max(3, n_thresholds + 1)
        max_cluster_count = min(15, n_thresholds * 2 + 2)
    else:
        min_clusters = 3
    
    from sklearn.metrics import silhouette_score
    
    best_cluster_count = min_clusters
    best_quality = -1.0
    
    for k in range(min_clusters, max_cluster_count + 1):
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(clustering_points)
        
        if len(set(cluster_labels)) > 1:
            try:
                quality = silhouette_score(clustering_points, cluster_labels)
                if quality > best_quality:
                    best_quality = quality
                    best_cluster_count = k
            except:
                pass
    
    final_clusterer = KMeans(
        n_clusters=best_cluster_count, 
        random_state=42, 
        n_init=10
    )
    final_clusterer.fit(clustering_points)
    cluster_centers = sorted(final_clusterer.cluster_centers_.flatten())
    
    data_span = data_max - data_min
    cutpoints = [data_min - data_span * 0.2]
    
    for i in range(len(cluster_centers) - 1):
        boundary = (cluster_centers[i] + cluster_centers[i + 1]) / 2
        cutpoints.append(boundary)
    
    cutpoints.append(data_max + data_span * 0.2)
    
    cutpoints = guarantee_complete_coverage(
        cutpoints, 
        rule_thresholds, 
        clean_data
    )
    
    return sorted(set(cutpoints))


def optimize_cutpoints_via_adaptive_percentiles(
    rule_thresholds: List[float],
    data_distribution: np.ndarray,
    base_percentile_list: Optional[List[int]] = None
) -> List[float]:
    """Generate cutpoints using adaptive percentile alignment.

    This function calculates base percentiles from data distribution, determines the
    percentile position of each threshold, adds neighboring percentiles within ±5%,
    filters cutpoints to maintain minimum separation of 3% of data span, and extends
    the range with 20% margin.

    Args:
        rule_thresholds (List[float]): Decision rule thresholds.
        data_distribution (np.ndarray): Clean numeric data array.
        base_percentile_list (Optional[List[int]]): Base percentiles to use,
                                                    auto-selected based on threshold
                                                    count if None.

    Returns:
        List[float]: Percentile-optimized cutpoints.
    """
    clean_data = data_distribution[~np.isnan(data_distribution)]
    
    if len(clean_data) < 10:
        return generate_fallback_cutpoints(rule_thresholds, clean_data)
    
    unique_thresholds = sorted(set(rule_thresholds))
    
    if base_percentile_list is None:
        n_thresholds = len(unique_thresholds)
        
        if n_thresholds <= 3:
            base_percentile_list = [10, 25, 50, 75, 90]
        elif n_thresholds <= 6:
            base_percentile_list = [5, 15, 30, 50, 70, 85, 95]
        else:
            base_percentile_list = [
                5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95
            ]
    
    base_cutpoints = np.percentile(clean_data, base_percentile_list)
    
    adaptive_cutpoints = []
    for threshold_val in unique_thresholds:
        percentile_position = (
            np.sum(clean_data <= threshold_val) / len(clean_data) * 100
        )
        adaptive_cutpoints.append(
            np.percentile(clean_data, percentile_position)
        )
        
        for offset in [-5, 5]:
            neighbor_percentile = np.clip(
                percentile_position + offset, 0, 100
            )
            adaptive_cutpoints.append(
                np.percentile(clean_data, neighbor_percentile)
            )
    
    all_cutpoints = sorted(set(
        list(base_cutpoints) + 
        adaptive_cutpoints + 
        unique_thresholds
    ))
    
    data_min, data_max = np.min(clean_data), np.max(clean_data)
    data_span = data_max - data_min
    min_separation = data_span * 0.03
    
    filtered_cutpoints = [all_cutpoints[0]]
    for cp in all_cutpoints[1:]:
        if cp - filtered_cutpoints[-1] >= min_separation:
            filtered_cutpoints.append(cp)
    
    range_extension = data_span * 0.2
    final_cutpoints = [
        data_min - range_extension,
        *filtered_cutpoints,
        data_max + range_extension
    ]
    
    final_cutpoints = guarantee_complete_coverage(
        final_cutpoints,
        rule_thresholds,
        clean_data
    )
    
    return sorted(set(final_cutpoints))


def optimize_cutpoints_via_constrained_optimization(
    rule_thresholds: List[float],
    data_distribution: np.ndarray,
    target_cutpoint_count: Optional[int] = None
) -> List[float]:
    """Generate cutpoints via constrained mathematical optimization.

    This function defines an objective function that minimizes coverage penalty
    (15× weight for distance from thresholds), density penalty (3× weight preferring
    high-density regions), and uniformity penalty (1× weight for even spacing).
    Constraints enforce monotonic ordering and bounds within data range.

    Args:
        rule_thresholds (List[float]): Decision rule thresholds.
        data_distribution (np.ndarray): Clean numeric data array.
        target_cutpoint_count (Optional[int]): Desired number of cutpoints,
                                               automatically determined if None.

    Returns:
        List[float]: Optimization-derived cutpoints.
    """
    clean_data = data_distribution[~np.isnan(data_distribution)]
    
    if len(clean_data) < 10:
        return generate_fallback_cutpoints(rule_thresholds, clean_data)
    
    unique_thresholds = sorted(set(rule_thresholds))
    data_min, data_max = np.min(clean_data), np.max(clean_data)
    
    if target_cutpoint_count is None:
        n_thresholds = len(unique_thresholds)
        target_cutpoint_count = int(
            min(12, max(n_thresholds + 2, n_thresholds * 1.5))
        )
    
    histogram_counts, bin_edges = np.histogram(clean_data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    density_profile = histogram_counts / histogram_counts.sum()
    
    def optimization_objective(internal_cutpoints):
        full_sequence = np.sort(np.concatenate([
            [data_min - 1],
            internal_cutpoints,
            [data_max + 1]
        ]))
        
        coverage_cost = sum(
            min(abs(full_sequence - threshold)) ** 2
            for threshold in unique_thresholds
        )
        
        density_cost = sum(
            (1 - np.interp(cp, bin_centers, density_profile)) ** 2
            for cp in internal_cutpoints
        )
        
        uniformity_cost = np.std(np.diff(full_sequence))
        
        return 15 * coverage_cost + 3 * density_cost + uniformity_cost
    
    def monotonicity_constraint(cutpoints):
        return np.diff(np.concatenate([
            [data_min],
            cutpoints,
            [data_max]
        ]))
    
    initial_guess = np.linspace(data_min, data_max, target_cutpoint_count)
    for i in range(min(len(unique_thresholds), target_cutpoint_count)):
        initial_guess[i] = unique_thresholds[i]
    initial_guess = np.sort(initial_guess)
    
    optimization_result = minimize(
        optimization_objective,
        initial_guess,
        method='SLSQP',
        bounds=[(data_min, data_max)] * target_cutpoint_count,
        constraints={
            'type': 'ineq',
            'fun': monotonicity_constraint
        },
        options={'maxiter': 200}
    )
    
    if optimization_result.success:
        optimized_cutpoints = sorted(optimization_result.x)
    else:
        optimized_cutpoints = sorted(initial_guess)
    
    data_span = data_max - data_min
    range_extension = data_span * 0.2
    
    final_cutpoints = [
        data_min - range_extension,
        *optimized_cutpoints,
        data_max + range_extension
    ]
    
    final_cutpoints = guarantee_complete_coverage(
        final_cutpoints,
        rule_thresholds,
        clean_data
    )
    
    return sorted(set(final_cutpoints))


def select_optimal_cutpoint_strategy(
    rule_thresholds: List[float],
    data_distribution: np.ndarray
) -> Tuple[str, List[float]]:
    """Auto-select best cutpoint optimization strategy.

    This function evaluates three strategies (density clustering, adaptive percentiles,
    and constrained optimization) using a composite quality score based on coverage
    (70% weight for threshold bracketing), size penalty (15% weight preferring fewer
    cutpoints), and uniformity score (15% weight for even spacing).

    Args:
        rule_thresholds (List[float]): Thresholds from decision rules.
        data_distribution (np.ndarray): Numeric data array.

    Returns:
        Tuple[str, List[float]]: A tuple containing the selected strategy name and
                                 the optimized cutpoints.
    """
    strategies = {
        'density_clustering': optimize_cutpoints_via_density_clustering,
        'adaptive_percentiles': optimize_cutpoints_via_adaptive_percentiles,
        'linear_programming': optimize_cutpoints_via_constrained_optimization
    }
    
    evaluation_results = {}
    
    for strategy_name, strategy_function in strategies.items():
        try:
            cutpoints = strategy_function(rule_thresholds, data_distribution)
            
            unique_thresholds = sorted(set(rule_thresholds))
            
            coverage_count = 0
            for threshold in unique_thresholds:
                has_lower_bracket = any(cp < threshold for cp in cutpoints)
                has_upper_bracket = any(cp > threshold for cp in cutpoints)
                
                if has_lower_bracket and has_upper_bracket:
                    coverage_count += 1
            
            coverage_score = (
                coverage_count / len(unique_thresholds)
                if unique_thresholds else 0
            )
            
            cutpoint_count = len(cutpoints)
            size_penalty = 1.0 / (1.0 + 0.1 * max(0, cutpoint_count - 8))
            
            if len(cutpoints) > 1:
                spacing_diffs = np.diff(sorted(cutpoints))
                uniformity_score = (
                    1.0 - (np.std(spacing_diffs) / np.mean(spacing_diffs))
                    if np.mean(spacing_diffs) > 0 else 0
                )
            else:
                uniformity_score = 0
            
            composite_quality = (
                coverage_score * 0.7 +
                size_penalty * 0.15 +
                uniformity_score * 0.15
            )
            
            evaluation_results[strategy_name] = {
                'cutpoints': cutpoints,
                'quality': composite_quality,
                'coverage': coverage_score,
                'count': cutpoint_count
            }
            
        except Exception as error:
            evaluation_results[strategy_name] = {
                'cutpoints': generate_fallback_cutpoints(
                    rule_thresholds, 
                    data_distribution
                ),
                'quality': 0.0,
                'coverage': 0.0,
                'count': 0
            }
    
    best_strategy_name, best_result = max(
        evaluation_results.items(),
        key=lambda item: item[1]['quality']
    )
    
    return best_strategy_name, best_result['cutpoints']


def generate_linguistic_terms(
    cutpoint_list: List[float]
) -> Dict[str, Tuple[float, float, float, float]]:
    """Create trapezoidal fuzzy membership functions from cutpoints.

    This function generates linguistic terms with adaptive design based on cutpoint
    count. Each term is defined by four parameters (a, b, c, d) representing support
    start, core start, core end, and support end respectively. The function handles
    edge cases for 3, 4, 5 cutpoints and dynamically generates up to 7 terms for
    6 or more cutpoints.

    Args:
        cutpoint_list (List[float]): Sorted cutpoint values.

    Returns:
        Dict[str, Tuple[float, float, float, float]]: Dictionary mapping linguistic
                                                       labels to trapezoid parameters
                                                       (a, b, c, d).
    """
    n_cutpoints = len(cutpoint_list)
    linguistic_terms = {}
    
    if n_cutpoints < 3:
        linguistic_terms['Medium'] = (
            cutpoint_list[0], 
            cutpoint_list[0],
            cutpoint_list[-1], 
            cutpoint_list[-1]
        )
        return linguistic_terms
    
    if n_cutpoints == 3:
        linguistic_terms['Low'] = (
            cutpoint_list[0], cutpoint_list[0],
            cutpoint_list[1], cutpoint_list[2]
        )
        linguistic_terms['High'] = (
            cutpoint_list[0], cutpoint_list[1],
            cutpoint_list[2], cutpoint_list[2]
        )
        return linguistic_terms
    
    if n_cutpoints == 4:
        linguistic_terms['VeryLow'] = (
            cutpoint_list[0], cutpoint_list[0],
            cutpoint_list[1], cutpoint_list[2]
        )
        linguistic_terms['Low'] = (
            cutpoint_list[0], cutpoint_list[1],
            cutpoint_list[2], cutpoint_list[3]
        )
        linguistic_terms['High'] = (
            cutpoint_list[1], cutpoint_list[2],
            cutpoint_list[3], cutpoint_list[3]
        )
        return linguistic_terms
    
    if n_cutpoints == 5:
        linguistic_terms['VeryLow'] = (
            cutpoint_list[0], cutpoint_list[0],
            cutpoint_list[1], cutpoint_list[2]
        )
        linguistic_terms['Low'] = (
            cutpoint_list[0], cutpoint_list[1],
            cutpoint_list[2], cutpoint_list[3]
        )
        linguistic_terms['Medium'] = (
            cutpoint_list[1], cutpoint_list[2],
            cutpoint_list[3], cutpoint_list[4]
        )
        linguistic_terms['High'] = (
            cutpoint_list[2], cutpoint_list[3],
            cutpoint_list[4], cutpoint_list[4]
        )
        return linguistic_terms
    
    n_terms_target = min(7, max(3, n_cutpoints - 2))
    
    term_label_pool = [
        'VeryLow', 'Low', 'MediumLow', 'Medium', 
        'MediumHigh', 'High', 'VeryHigh'
    ]
    selected_labels = term_label_pool[:n_terms_target]
    
    for term_index, term_label in enumerate(selected_labels):
        if term_index == 0:
            idx_a = 0
            idx_b = 0
            idx_c = min(n_cutpoints - 1, max(1, int((n_cutpoints - 1) * 0.2)))
            idx_d = min(n_cutpoints - 1, max(2, int((n_cutpoints - 1) * 0.4)))
        
        elif term_index == n_terms_target - 1:
            idx_a = max(0, min(n_cutpoints - 3, int((n_cutpoints - 1) * 0.6)))
            idx_b = max(0, min(n_cutpoints - 2, int((n_cutpoints - 1) * 0.8)))
            idx_c = n_cutpoints - 1
            idx_d = n_cutpoints - 1
        
        else:
            relative_position = (term_index + 0.5) / n_terms_target
            term_width = 1.0 / n_terms_target
            
            idx_a = max(0, min(n_cutpoints - 1, 
                int((n_cutpoints - 1) * (relative_position - term_width))))
            idx_b = max(0, min(n_cutpoints - 1,
                int((n_cutpoints - 1) * (relative_position - term_width / 2))))
            idx_c = max(0, min(n_cutpoints - 1,
                int((n_cutpoints - 1) * (relative_position + term_width / 2))))
            idx_d = max(0, min(n_cutpoints - 1,
                int((n_cutpoints - 1) * (relative_position + term_width))))
        
        idx_a = min(idx_a, n_cutpoints - 1)
        idx_b = max(idx_a, min(idx_b, n_cutpoints - 1))
        idx_c = max(idx_b, min(idx_c, n_cutpoints - 1))
        idx_d = max(idx_c, min(idx_d, n_cutpoints - 1))
        
        linguistic_terms[term_label] = (
            cutpoint_list[idx_a],
            cutpoint_list[idx_b],
            cutpoint_list[idx_c],
            cutpoint_list[idx_d]
        )
    
    return linguistic_terms


def generate_binary_linguistic_terms() -> Dict[str, Tuple[float, float, float, float]]:
    """Create fuzzy terms for binary variables.

    This function generates two overlapping trapezoidal membership functions for
    binary variables, with Low term centered on 0 and High term centered on 1.

    Returns:
        Dict[str, Tuple[float, float, float, float]]: Dictionary containing Low and
                                                       High linguistic terms with
                                                       trapezoid parameters.
    """
    return {
        'Low': (-0.1, -0.1, 0.5, 0.6),
        'High': (0.4, 0.5, 1.1, 1.1)
    }


def parse_decision_condition(
    condition_text: str
) -> Tuple[str, str, Any]:
    """Extract variable, operator, and value from condition string.

    This function parses numeric condition strings to extract the variable name,
    comparison operator, and threshold value. The function handles conditions for
    both continuous numeric features and encoded categorical features treated as
    numeric.

    Args:
        condition_text (str): Raw condition string in format "variable op value".

    Returns:
        Tuple[str, str, Any]: A tuple containing (variable_name, operator,
                             threshold_value).

    Raises:
        ValueError: If the condition string cannot be parsed.
    """
    condition_text = condition_text.strip()
    
    numeric_pattern = r'(.+?)\s*(<=|>=|<|>|==|=)\s*(-?\d+\.?\d*)'
    numeric_match = re.match(numeric_pattern, condition_text)
    
    if numeric_match:
        variable_name = numeric_match.group(1).strip()
        operator = numeric_match.group(2).strip()
        value = float(numeric_match.group(3).strip())
        return variable_name, operator, value
    
    raise ValueError(f"Cannot parse condition: {condition_text}")


def extract_conditions_from_rules(
    decision_rules: List[Dict]
) -> Dict[str, List[Tuple[str, Any]]]:
    """Extract all variable conditions from rule set.

    This function processes a list of decision rules to extract all conditions,
    parsing each condition string to identify variables and their associated
    operators and threshold values.

    Args:
        decision_rules (List[Dict]): List of decision rule dictionaries containing
                                     condition information.

    Returns:
        Dict[str, List[Tuple[str, Any]]]: Dictionary mapping variable names to
                                          lists of (operator, value) tuples.
    """
    variable_condition_map = {}
    
    for rule_dict in decision_rules:
        condition_list = rule_dict.get(
            'rule_conditions', 
            rule_dict.get('conditions', [])
        )
        
        for condition_item in condition_list:
            if isinstance(condition_item, str):
                condition_string = condition_item
            elif isinstance(condition_item, dict):
                feature = condition_item.get('feature', '')
                operator = condition_item.get('operator', '')
                threshold = condition_item.get('threshold', 0)
                condition_string = f"{feature} {operator} {threshold}"
            else:
                continue
            
            try:
                variable, operator, value = parse_decision_condition(
                    condition_string
                )
                
                if variable not in variable_condition_map:
                    variable_condition_map[variable] = []
                
                variable_condition_map[variable].append((operator, value))
                
            except ValueError:
                continue
    
    return variable_condition_map


def decode_categorical_condition(
    variable_name: str,
    operator: str,
    threshold_value: float,
    category_mappings: Dict[str, Dict]
) -> str:
    """Decode numeric categorical condition to human-readable format.

    This function converts encoded categorical conditions to their original categorical
    representations. The function handles various comparison operators and safely
    processes type conversions and edge cases.

    Args:
        variable_name (str): Feature name.
        operator (str): Comparison operator (<=, <, >, >=, ==, =).
        threshold_value (float): Numeric threshold from encoded condition.
        category_mappings (Dict[str, Dict]): Mappings in format {feature: {code: category}}.

    Returns:
        str: Decoded condition string in human-readable format, or original condition
             if decoding fails.
    """
    if variable_name not in category_mappings:
        return f"{variable_name} {operator} {threshold_value}"
    
    mapping = category_mappings[variable_name]
    
    try:
        code = int(round(threshold_value))
    except (ValueError, TypeError):
        return f"{variable_name} {operator} {threshold_value}"
    
    try:
        all_codes = sorted([int(k) for k in mapping.keys()])
    except (ValueError, TypeError):
        return f"{variable_name} {operator} {threshold_value}"
    
    if not all_codes:
        return f"{variable_name} {operator} {threshold_value}"
    
    min_code = min(all_codes)
    max_code = max(all_codes)
    
    if operator in ['<=', '<']:
        selected_codes = [c for c in all_codes if c <= code]
        
        if not selected_codes:
            return f"{variable_name} < {mapping.get(min_code, 'UNKNOWN')}"
        
        categories = [mapping.get(c, f'UNKNOWN_{c}') for c in selected_codes]
        
        if len(categories) == 1:
            return f"{variable_name} == '{categories[0]}'"
        else:
            return f"{variable_name} in {categories}"
    
    elif operator in ['>', '>=']:
        if operator == '>':
            selected_codes = [c for c in all_codes if c > code]
        else:
            selected_codes = [c for c in all_codes if c >= code]
        
        if not selected_codes:
            return f"{variable_name} > {mapping.get(max_code, 'UNKNOWN')}"
        
        categories = [mapping.get(c, f'UNKNOWN_{c}') for c in selected_codes]
        
        if len(categories) == 1:
            return f"{variable_name} == '{categories[0]}'"
        else:
            return f"{variable_name} in {categories}"
    
    elif operator in ['==', '=']:
        if code in mapping:
            return f"{variable_name} == '{mapping[code]}'"
        else:
            return f"{variable_name} == UNKNOWN_{code}"
    
    return f"{variable_name} {operator} {threshold_value}"


def transform_crisp_rule_to_fuzzy(
    crisp_conditions: List[str],
    variable_cutpoints: Dict[str, List[float]],
    linguistic_terms: Dict[str, Dict[str, Tuple]],
    category_mappings: Optional[Dict[str, Dict]] = None
) -> Dict:
    """Convert crisp decision rule to fuzzy representation.

    This function transforms crisp rule conditions into fuzzy antecedents by parsing
    each condition, determining appropriate linguistic terms based on operator and
    threshold, and decoding categorical conditions to human-readable format when
    applicable.

    Args:
        crisp_conditions (List[str]): List of crisp condition strings.
        variable_cutpoints (Dict[str, List[float]]): Cutpoint sets per variable.
        linguistic_terms (Dict[str, Dict[str, Tuple]]): Linguistic term definitions
                                                        per variable.
        category_mappings (Optional[Dict[str, Dict]]): Category mappings for decoding
                                                       categorical conditions.

    Returns:
        Dict: Dictionary containing fuzzy antecedents with linguistic terms, original
              and decoded conditions, and membership parameters.
    """
    if category_mappings is None:
        category_mappings = {}
    
    fuzzy_antecedents = []
    
    for condition_text in crisp_conditions:
        try:
            variable_name, operator, threshold_value = parse_decision_condition(
                condition_text
            )
        except ValueError:
            continue
        
        clean_variable = variable_name.replace("Unnamed: ", "").strip()
        
        if clean_variable not in linguistic_terms:
            continue
        
        term_dict = linguistic_terms[clean_variable]
        
        term_center_map = {}
        for term_name, (a, b, c, d) in term_dict.items():
            term_center_map[term_name] = (b + c) / 2
        
        selected_term_list = []
        
        if operator in ['<', '<=']:
            for term_name, center_value in term_center_map.items():
                if center_value <= threshold_value:
                    selected_term_list.append(term_name)
        
        elif operator in ['>', '>=']:
            for term_name, center_value in term_center_map.items():
                if center_value >= threshold_value:
                    selected_term_list.append(term_name)
        
        elif operator in ['==', '=']:
            closest_term = min(
                term_center_map.keys(),
                key=lambda t: abs(term_center_map[t] - threshold_value)
            )
            selected_term_list.append(closest_term)
        
        selected_term_list.sort(key=lambda t: term_center_map[t])
        
        is_categorical = clean_variable in category_mappings
        
        if is_categorical:
            try:
                decoded_condition = decode_categorical_condition(
                    clean_variable,
                    operator,
                    threshold_value,
                    category_mappings
                )
            except Exception as e:
                decoded_condition = condition_text
                is_categorical = False
        else:
            decoded_condition = condition_text
        
        if selected_term_list:
            fuzzy_antecedents.append({
                'variable': clean_variable,
                'linguistic_terms': selected_term_list,
                'original_condition': condition_text,
                'decoded_condition': decoded_condition,
                'is_categorical': is_categorical,
                'membership_params': {
                    term: term_dict[term] 
                    for term in selected_term_list
                }
            })
    
    return {
        'antecedents': fuzzy_antecedents,
        'condition_count': len(fuzzy_antecedents)
    }


def transform_all_rules_to_fuzzy(
    crisp_rules: List[Dict],
    variable_cutpoints: Dict[str, List[float]],
    linguistic_terms: Dict[str, Dict[str, Tuple]],
    category_mappings: Optional[Dict[str, Dict]] = None
) -> List[Dict]:
    """Batch transform all crisp rules to fuzzy format.

    This function processes a list of crisp decision rules and transforms each into
    fuzzy representation while preserving metadata such as tree identifier, logit
    contribution, and predicted class.

    Args:
        crisp_rules (List[Dict]): Original decision rules with crisp conditions.
        variable_cutpoints (Dict[str, List[float]]): Cutpoint sets per variable.
        linguistic_terms (Dict[str, Dict[str, Tuple]]): Fuzzy term definitions.
        category_mappings (Optional[Dict[str, Dict]]): Categorical feature mappings.

    Returns:
        List[Dict]: List of fuzzy rule dictionaries with antecedents and metadata.
    """
    fuzzy_rule_list = []
    
    for rule_index, crisp_rule in enumerate(crisp_rules):
        condition_list = crisp_rule.get(
            'rule_conditions',
            crisp_rule.get('conditions', [])
        )
        tree_identifier = crisp_rule.get(
            'tree_idx',
            crisp_rule.get('tree_id', -1)
        )
        
        logit_contribution = crisp_rule.get('logit_score')
        predicted_class = crisp_rule.get('predicted_class')
        
        fuzzy_representation = transform_crisp_rule_to_fuzzy(
            condition_list,
            variable_cutpoints,
            linguistic_terms,
            category_mappings
        )
        
        fuzzy_representation['rule_id'] = rule_index
        fuzzy_representation['tree_id'] = tree_identifier
        fuzzy_representation['logit_score'] = logit_contribution
        fuzzy_representation['predicted_class'] = predicted_class
        
        fuzzy_rule_list.append(fuzzy_representation)
    
    return fuzzy_rule_list


def compute_trapezoidal_membership(
    crisp_value: float,
    trapezoid_a: float,
    trapezoid_b: float,
    trapezoid_c: float,
    trapezoid_d: float
) -> float:
    """Calculate trapezoidal fuzzy membership degree.

    This function computes the membership degree for a crisp value using trapezoidal
    membership function with four parameters. The function supports infinite extension
    for edge terms: left-extended terms have membership 1.0 for values at or below
    core start, and right-extended terms have membership 1.0 for values at or above
    core start.

    Args:
        crisp_value (float): Input value to fuzzify.
        trapezoid_a (float): Support start parameter.
        trapezoid_b (float): Core start parameter.
        trapezoid_c (float): Core end parameter.
        trapezoid_d (float): Support end parameter.

    Returns:
        float: Membership degree in range [0, 1].
    """
    if (np.isinf(trapezoid_a) and trapezoid_a < 0 and
        np.isinf(trapezoid_b) and trapezoid_b < 0):
        
        if crisp_value <= trapezoid_c:
            return 1.0
        elif crisp_value < trapezoid_d:
            return (
                (trapezoid_d - crisp_value) / (trapezoid_d - trapezoid_c)
                if trapezoid_d > trapezoid_c else 0.0
            )
        else:
            return 0.0
    
    if (np.isinf(trapezoid_c) and trapezoid_c > 0 and
        np.isinf(trapezoid_d) and trapezoid_d > 0):
        
        if crisp_value >= trapezoid_b:
            return 1.0
        elif crisp_value > trapezoid_a:
            return (
                (crisp_value - trapezoid_a) / (trapezoid_b - trapezoid_a)
                if trapezoid_b > trapezoid_a else 0.0
            )
        else:
            return 0.0
    
    if crisp_value <= trapezoid_a or crisp_value >= trapezoid_d:
        return 0.0
    
    elif trapezoid_a < crisp_value <= trapezoid_b:
        return (
            (crisp_value - trapezoid_a) / (trapezoid_b - trapezoid_a)
            if trapezoid_b > trapezoid_a else 1.0
        )
    
    elif trapezoid_b < crisp_value < trapezoid_c:
        return 1.0
    
    else:
        return (
            (trapezoid_d - crisp_value) / (trapezoid_d - trapezoid_c)
            if trapezoid_d > trapezoid_c else 1.0
        )


def fuzzify_crisp_value(
    crisp_value: float,
    linguistic_term_dict: Dict[str, Tuple]
) -> Dict[str, float]:
    """Convert crisp value to fuzzy membership vector.

    This function fuzzifies a numeric input value by computing membership degrees
    for all defined linguistic terms using trapezoidal membership functions.

    Args:
        crisp_value (float): Numeric input value.
        linguistic_term_dict (Dict[str, Tuple]): Fuzzy term definitions mapping term
                                                 names to trapezoid parameters.

    Returns:
        Dict[str, float]: Dictionary mapping term names to membership degrees.
    """
    membership_vector = {}
    
    for term_label, (a, b, c, d) in linguistic_term_dict.items():
        membership_vector[term_label] = compute_trapezoidal_membership(
            crisp_value, a, b, c, d
        )
    
    return membership_vector


def compute_rule_activation(
    data_row: pd.Series,
    fuzzy_rule: Dict,
    linguistic_term_library: Dict[str, Dict],
    category_mappings: Optional[Dict[str, Dict]] = None,
    t_norm_operator: str = 'min'
) -> float:
    """Calculate activation degree for fuzzy rule.

    This function computes rule activation by fuzzifying each variable in the rule
    antecedents, taking the maximum membership across selected terms per variable,
    and aggregating using either minimum (Gödel) or product (Probabilistic) T-norm.
    All variables are treated as numeric since categorical features are pre-encoded.

    Args:
        data_row (pd.Series): Input data sample with encoded features.
        fuzzy_rule (Dict): Fuzzy rule dictionary with antecedents.
        linguistic_term_library (Dict[str, Dict]): Term definitions per variable.
        category_mappings (Optional[Dict[str, Dict]]): Not used, kept for compatibility.
        t_norm_operator (str): Aggregation operator, 'min' for Gödel T-norm or
                              'product' for Probabilistic T-norm.

    Returns:
        float: Rule activation degree in range [0, 1].
    """
    if category_mappings is None:
        category_mappings = {}
    
    activation_degree_list = []
    
    for antecedent in fuzzy_rule['antecedents']:
        variable_name = antecedent['variable']
        selected_term_list = antecedent['linguistic_terms']
        
        if variable_name not in data_row.index or pd.isna(data_row[variable_name]):
            activation_degree_list.append(0.0)
            continue
        
        if variable_name not in linguistic_term_library:
            continue
        
        raw_value = data_row[variable_name]
        
        try:
            numeric_value = float(raw_value)
        except (ValueError, TypeError):
            activation_degree_list.append(0.0)
            continue
        
        membership_degrees = fuzzify_crisp_value(
            numeric_value,
            linguistic_term_library[variable_name]
        )
        
        max_membership = 0.0
        for term_name in selected_term_list:
            term_membership = membership_degrees.get(term_name, 0.0)
            max_membership = max(max_membership, term_membership)
        
        activation_degree_list.append(max_membership)
    
    if not activation_degree_list:
        return 0.0
    
    if t_norm_operator == 'product':
        aggregated_activation = 1.0
        for degree in activation_degree_list:
            aggregated_activation *= degree
        return aggregated_activation
    else:
        return min(activation_degree_list)


def execute_fuzzy_inference(
    feature_matrix: pd.DataFrame,
    fuzzy_rule_set: List[Dict],
    linguistic_term_library: Dict[str, Dict],
    category_mappings: Optional[Dict[str, Dict]] = None,
    t_norm_operator: str = 'min',
    classification_threshold: float = 0.45,
    parallel_jobs: int = -1,
    batch_size: int = 100
) -> np.ndarray:
    """Execute fuzzy inference using all pre-selected trees.

    This function performs fuzzy inference over the feature matrix using fuzzy rules
    that have been pre-selected by select_best_trees. The function eliminates the
    redundant top-K selection step and uses all provided trees, applying single
    normalization at tree level without additional normalization of cumulative logits.

    Args:
        feature_matrix (pd.DataFrame): DataFrame with encoded features.
        fuzzy_rule_set (List[Dict]): List of fuzzy rules pre-filtered by select_best_trees.
        linguistic_term_library (Dict[str, Dict]): Linguistic terms per variable.
        category_mappings (Optional[Dict[str, Dict]]): Category mappings, optional.
        t_norm_operator (str): Aggregation operator, 'min' or 'product'.
        classification_threshold (float): Threshold for binary classification.
        parallel_jobs (int): Number of CPUs for parallelization, -1 for all.
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Binary predictions array.
    """
    if category_mappings is None:
        category_mappings = {}
    
    import time
    import numpy as np
    from collections import defaultdict
    from joblib import Parallel, delayed
    
    start_time = time.time()
    
    print(f"\n[INFO] Preparing optimized inference")
    
    tree_rule_groups = defaultdict(list)
    for rule in fuzzy_rule_set:
        if 'tree_id' in rule:
            tree_id = rule['tree_id']
        elif 'tree_idx' in rule:
            tree_id = rule['tree_idx']
        else:
            tree_id = rule.get('tree_index', 0)
        
        tree_rule_groups[tree_id].append(rule)
    
    total_trees = len(tree_rule_groups)
    
    tree_data = {}
    for tree_id, rules in tree_rule_groups.items():
        rule_data = []
        for rule in rules:
            antecedents_processed = []
            for ant in rule['antecedents']:
                antecedents_processed.append({
                    'var': ant['variable'],
                    'terms': tuple(ant['linguistic_terms'])
                })
            
            logit_val = rule.get('logit', rule.get('logit_score', 0.0))
            
            rule_data.append({
                'antecedents': antecedents_processed,
                'logit': logit_val
            })
        
        tree_data[tree_id] = rule_data
    
    feature_names = feature_matrix.columns.tolist()
    feature_values = feature_matrix.values
    
    print(f"[INFO] Total trees: {total_trees}")
    print(f"[INFO] Total rules: {len(fuzzy_rule_set):,}")
    print(f"[INFO] Samples: {len(feature_matrix):,}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Using ALL trees (pre-selected by select_best_trees)")
    
    def predict_batch(batch_indices):
        """Predict batch of samples using all fuzzy trees."""
        batch_predictions = np.zeros(len(batch_indices), dtype=np.int32)
        
        for batch_idx, sample_idx in enumerate(batch_indices):
            row_values = feature_values[sample_idx]
            
            membership_cache = {}
            
            for feat_idx, var_name in enumerate(feature_names):
                if var_name not in linguistic_term_library:
                    continue
                
                value = row_values[feat_idx]
                
                if pd.isna(value):
                    continue
                
                try:
                    numeric_value = float(value)
                    membership_cache[var_name] = fuzzify_crisp_value(
                        numeric_value,
                        linguistic_term_library[var_name]
                    )
                except (ValueError, TypeError):
                    continue
            
            tree_scores = []
            
            for tree_id, rules in tree_data.items():
                tree_logit = 0.0
                tree_activation = 0.0
                n_rules = len(rules)
                
                for rule in rules:
                    activation_degrees = []
                    
                    for ant in rule['antecedents']:
                        var_name = ant['var']
                        
                        if var_name not in membership_cache:
                            activation_degrees.append(0.0)
                            continue
                        
                        max_membership = 0.0
                        memberships = membership_cache[var_name]
                        
                        for term in ant['terms']:
                            mem = memberships.get(term, 0.0)
                            if mem > max_membership:
                                max_membership = mem
                        
                        activation_degrees.append(max_membership)
                    
                    if not activation_degrees:
                        continue
                    
                    if t_norm_operator == 'product':
                        rule_activation = np.prod(activation_degrees)
                    else:
                        rule_activation = min(activation_degrees)
                    
                    tree_logit += rule_activation * rule['logit']
                    tree_activation += rule_activation
                
                if tree_activation > 1e-10:
                    normalized_logit = tree_logit / tree_activation
                    
                    tree_scores.append({
                        'logit': normalized_logit
                    })
            
            if not tree_scores:
                batch_predictions[batch_idx] = 0
                continue
            
            logits = np.array([t['logit'] for t in tree_scores])
            cumulative_logit = logits.sum()
            
            probability = 1.0 / (1.0 + np.exp(-cumulative_logit))
            batch_predictions[batch_idx] = 1 if probability >= classification_threshold else 0
        
        return batch_predictions
    
    n_samples = len(feature_matrix)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = list(range(i, min(i + batch_size, n_samples)))
        batches.append(batch_indices)
    
    n_jobs = parallel_jobs if parallel_jobs > 0 else -1
    
    print(f"[INFO] Batches: {len(batches)}")
    print(f"[INFO] Parallel jobs: {n_jobs}")
    print(f"\n[INFO] Starting inference")
    
    batch_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(predict_batch)(batch)
        for batch in batches
    )
    
    predictions = np.concatenate(batch_results)
    
    elapsed = time.time() - start_time
    throughput = n_samples / elapsed
    
    print(f"\n[INFO] Inference completed in {elapsed:.1f}s")
    print(f"[INFO] Throughput: {throughput:.1f} samples/sec")
    
    return predictions


def evaluate_fuzzy_system(
    catboost_rules: List[Dict],
    catboost_predictions: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    category_mappings: Optional[Dict[str, Dict]] = None,
    classification_threshold: float = 0.65,
    t_norm_operator: str = 'min',
    evaluation_sample_size: int = 50000,
    parallel_jobs: int = -1,
    apply_purpose_grouping: bool = True
) -> Tuple[Dict, Dict, List[Dict], Dict]:
    """Execute complete fuzzy system evaluation pipeline.

    This function performs a comprehensive evaluation workflow including optional loan
    purpose grouping, variable condition extraction, automatic cutpoint optimization
    per variable, linguistic term generation, fuzzy rule transformation with categorical
    decoding, fuzzy inference execution, performance metric calculation, and diagnostic
    report generation. All features are treated as numeric since categorical features
    are pre-encoded.

    Args:
        catboost_rules (List[Dict]): Extracted decision rules with numeric conditions.
        catboost_predictions (np.ndarray): CatBoost model predictions.
        X_train (pd.DataFrame): Training features, all encoded as numeric.
        X_test (pd.DataFrame): Test features, all encoded as numeric.
        y_test (np.ndarray): Test labels.
        category_mappings (Optional[Dict[str, Dict]]): Mappings in format
                                                       {feature: {category: code}}.
        classification_threshold (float): Probability threshold for classification.
        t_norm_operator (str): T-norm aggregation operator.
        evaluation_sample_size (int): Maximum samples for evaluation.
        parallel_jobs (int): Number of CPU cores for parallel processing.
        apply_purpose_grouping (bool): Whether to apply purpose category grouping.

    Returns:
        Tuple[Dict, Dict, List[Dict], Dict]: A tuple containing variable_cutpoints,
                                             linguistic_terms, fuzzy_rules, and
                                             evaluation_results dictionaries.
    """
    print("\n[INFO] -------------------- Fuzzy CatBoost System Evaluation --------------------")
    print(f"Classification threshold: {classification_threshold}")
    print(f"T-norm operator: {t_norm_operator}")
    print(f"Purpose grouping: {apply_purpose_grouping}")
    print(f"All features treated as numeric (categorical pre-encoded)")
    print("[INFO] -------------------------------------------------------------------------")
    
    if category_mappings is None:
        category_mappings = {}
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    category_mappings_copy = category_mappings.copy()
    
    if apply_purpose_grouping and 'purpose' in X_train_copy.columns:
        print("\n[INFO] Applying purpose category grouping")
        
        if 'purpose' in category_mappings_copy:
            purpose_mapping = category_mappings_copy['purpose']
            
            purpose_groups = define_loan_purpose_groups()
            
            reverse_map = {code: cat for cat, code in purpose_mapping.items()}
            
            new_purpose_mapping = {}
            for group_name, category_list in purpose_groups.items():
                group_codes = [
                    code for code, cat in reverse_map.items()
                    if cat in category_list
                ]
                for code in group_codes:
                    new_purpose_mapping[group_name] = code
                    break
            
            category_mappings_copy['purpose'] = new_purpose_mapping
            
            print(f"[INFO] Groups: {list(purpose_groups.keys())}")
    
    print("\n[INFO] Step 1: Extracting variable conditions from rules")
    
    variable_conditions = extract_conditions_from_rules(catboost_rules)
    
    print(f"[INFO] Rules processed: {len(catboost_rules)}")
    print(f"[INFO] Variables identified: {len(variable_conditions)}")
    
    print("\n[INFO] Step 2: Optimizing cutpoints with auto-strategy selection")
    
    variable_cutpoint_sets = {}
    linguistic_term_sets = {}
    optimization_methods_used = {}
    
    for variable_name, condition_tuples in variable_conditions.items():
        cleaned_variable = variable_name.replace("Unnamed: ", "")
        
        if cleaned_variable not in X_train_copy.columns:
            continue
        
        data_column = X_train_copy[cleaned_variable]
        
        variable_type = classify_variable_type(
            data_column,
            category_mappings_copy,
            cleaned_variable
        )
        
        print(f"[INFO] {cleaned_variable}: {variable_type}", end='')
        
        if variable_type == 'binary':
            variable_cutpoint_sets[cleaned_variable] = [-0.1, 0.5, 1.1]
            linguistic_term_sets[cleaned_variable] = generate_binary_linguistic_terms()
            optimization_methods_used[cleaned_variable] = 'binary'
            
            print(f" -> binary")
        
        else:
            threshold_values = [
                value 
                for operator, value in condition_tuples 
                if isinstance(value, (int, float))
            ]
            
            if not threshold_values:
                print(f" -> SKIP (no numeric thresholds)")
                continue
            
            best_strategy, optimized_cutpoints = select_optimal_cutpoint_strategy(
                threshold_values,
                data_column.values
            )
            
            variable_cutpoint_sets[cleaned_variable] = optimized_cutpoints
            linguistic_term_sets[cleaned_variable] = generate_linguistic_terms(
                optimized_cutpoints
            )
            optimization_methods_used[cleaned_variable] = best_strategy
            
            term_labels = list(linguistic_term_sets[cleaned_variable].keys())
            cat_label = " [CATEGORICAL]" if variable_type == 'categorical' else ""
            
            print(f"{cat_label} -> {best_strategy[:8]}, "
                  f"{len(optimized_cutpoints)} cutpoints, {len(term_labels)} terms")
    
    print(f"\n[INFO] Total variables processed: {len(linguistic_term_sets)}")
    
    print("\n[INFO] -------------------- Cutpoint Optimization Summary --------------------")
    
    method_groups = {}
    for var, method in optimization_methods_used.items():
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(var)
    
    method_full_names = {
        'density_clustering': 'Density-Weighted K-Means Clustering',
        'adaptive_percentiles': 'Adaptive Percentile Alignment',
        'linear_programming': 'Constrained Mathematical Programming',
        'binary': 'Binary Variable (Fixed)'
    }
    
    print(f"\n{'Method':<45} {'Variables':<12} {'Percentage'}")
    print("-" * 70)
    
    total_vars = len(optimization_methods_used)
    
    for method, variables in sorted(method_groups.items(), key=lambda x: len(x[1]), reverse=True):
        full_name = method_full_names.get(method, method)
        count = len(variables)
        percentage = (count / total_vars * 100) if total_vars > 0 else 0
        
        print(f"{full_name:<45} {count:<12} {percentage:>6.1f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<45} {total_vars:<12} {'100.0%':>7}")
    
    print("\n[INFO] -------------------- Variables by Optimization Method --------------------")
    
    for method, variables in sorted(method_groups.items(), key=lambda x: len(x[1]), reverse=True):
        full_name = method_full_names.get(method, method)
        
        print(f"\n{full_name} ({len(variables)} variables):")
        print("-" * 70)
        
        sorted_vars = sorted(variables)
        
        for i in range(0, len(sorted_vars), 3):
            row_vars = sorted_vars[i:i+3]
            row_str = "  ".join(f"{v:<25}" for v in row_vars)
            print(f"  {row_str}")
    
    print("\n[INFO] -------------------- Cutpoint and Linguistic Term Statistics --------------------")
    
    cutpoint_stats = []
    term_stats = []
    
    for var, method in optimization_methods_used.items():
        n_cutpoints = len(variable_cutpoint_sets.get(var, []))
        n_terms = len(linguistic_term_sets.get(var, {}))
        
        cutpoint_stats.append(n_cutpoints)
        term_stats.append(n_terms)
    
    print(f"\nCutpoints per variable:")
    print(f"  Min:     {min(cutpoint_stats) if cutpoint_stats else 0}")
    print(f"  Max:     {max(cutpoint_stats) if cutpoint_stats else 0}")
    print(f"  Mean:    {np.mean(cutpoint_stats) if cutpoint_stats else 0:.1f}")
    print(f"  Median:  {np.median(cutpoint_stats) if cutpoint_stats else 0:.1f}")
    
    print(f"\nLinguistic terms per variable:")
    print(f"  Min:     {min(term_stats) if term_stats else 0}")
    print(f"  Max:     {max(term_stats) if term_stats else 0}")
    print(f"  Mean:    {np.mean(term_stats) if term_stats else 0:.1f}")
    print(f"  Median:  {np.median(term_stats) if term_stats else 0:.1f}")
    
    term_distribution = Counter(term_stats)
    
    print(f"\nLinguistic term distribution:")
    print(f"{'# Terms':<12} {'# Variables':<15} {'Percentage'}")
    print("-" * 45)
    
    for n_terms in sorted(term_distribution.keys()):
        count = term_distribution[n_terms]
        percentage = (count / total_vars * 100) if total_vars > 0 else 0
        print(f"{n_terms:<12} {count:<15} {percentage:>6.1f}%")
    
    print("[INFO] -------------------------------------------------------------------------")
    
    print("\n[INFO] Step 3: Transforming rules to fuzzy format")
    
    fuzzy_rule_set = transform_all_rules_to_fuzzy(
        catboost_rules,
        variable_cutpoint_sets,
        linguistic_term_sets,
        category_mappings_copy
    )
    
    print(f"[INFO] Fuzzy rules created: {len(fuzzy_rule_set)}")
    
    categorical_conditions = sum(
        1 for rule in fuzzy_rule_set
        for ant in rule['antecedents']
        if ant.get('is_categorical', False)
    )
    
    print(f"[INFO] Categorical conditions decoded: {categorical_conditions}")
    
    if len(X_test_copy) > evaluation_sample_size:
        print(f"\n[INFO] Step 4: Sampling evaluation set ({evaluation_sample_size:,})")
        
        sampler = StratifiedShuffleSplit(
            n_splits=1,
            test_size=evaluation_sample_size,
            random_state=42
        )
        _, sample_indices = next(sampler.split(X_test_copy, y_test))
        
        X_eval = X_test_copy.iloc[sample_indices]
        y_eval = (
            y_test[sample_indices] 
            if isinstance(y_test, np.ndarray) 
            else y_test.iloc[sample_indices]
        )
        catboost_eval = (
            catboost_predictions[sample_indices]
            if isinstance(catboost_predictions, np.ndarray)
            else catboost_predictions.iloc[sample_indices]
        )
    else:
        X_eval = X_test_copy
        y_eval = y_test
        catboost_eval = catboost_predictions
    
    print(f"\n[INFO] Step 5: Executing fuzzy inference (parallel_jobs={parallel_jobs})")
    
    fuzzy_predictions = execute_fuzzy_inference(
        X_eval,
        fuzzy_rule_set,
        linguistic_term_sets,
        category_mappings_copy,
        t_norm_operator=t_norm_operator,
        classification_threshold=classification_threshold,
        parallel_jobs=parallel_jobs
    )
    
    print("\n[INFO] Step 6: Calculating performance metrics")
    
    fuzzy_performance = {
        'accuracy': accuracy_score(y_eval, fuzzy_predictions),
        'sensitivity': recall_score(y_eval, fuzzy_predictions, pos_label=1, zero_division=0),
        'precision': precision_score(y_eval, fuzzy_predictions, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_eval, fuzzy_predictions, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(y_eval, fuzzy_predictions),
        'predictions': fuzzy_predictions
    }
    
    catboost_performance = {
        'accuracy': accuracy_score(y_eval, catboost_eval),
        'sensitivity': recall_score(y_eval, catboost_eval, pos_label=1, zero_division=0),
        'precision': precision_score(y_eval, catboost_eval, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_eval, catboost_eval, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(y_eval, catboost_eval)
    }
    
    print("\n[INFO] -------------------- Evaluation Results --------------------")
    
    print(f"\n{'Metric':<15} {'Fuzzy':<10} {'CatBoost':<10} {'Difference':<12} {'% Change'}")
    print("-" * 60)
    
    for metric_label in ['accuracy', 'sensitivity', 'precision', 'f1_score']:
        fuzzy_value = fuzzy_performance[metric_label]
        catboost_value = catboost_performance[metric_label]
        difference = fuzzy_value - catboost_value
        percent_change = (
            (difference / catboost_value * 100)
            if catboost_value > 0 else 0
        )
        
        print(f"{metric_label.capitalize():<15} {fuzzy_value:<10.4f} "
              f"{catboost_value:<10.4f} {difference:+12.4f} {percent_change:+9.1f}%")
    
    print("\n[INFO] ----------------------------------------------------------------")
    
    evaluation_summary = {
        'fuzzy_metrics': fuzzy_performance,
        'catboost_metrics': catboost_performance,
        'optimization_methods': optimization_methods_used,
        'optimization_method_summary': method_groups,
        'cutpoint_statistics': {
            'min': min(cutpoint_stats) if cutpoint_stats else 0,
            'max': max(cutpoint_stats) if cutpoint_stats else 0,
            'mean': float(np.mean(cutpoint_stats)) if cutpoint_stats else 0,
            'median': float(np.median(cutpoint_stats)) if cutpoint_stats else 0
        },
        'linguistic_term_statistics': {
            'min': min(term_stats) if term_stats else 0,
            'max': max(term_stats) if term_stats else 0,
            'mean': float(np.mean(term_stats)) if term_stats else 0,
            'median': float(np.median(term_stats)) if term_stats else 0,
            'distribution': dict(term_distribution)
        },
        'classification_threshold': classification_threshold,
        't_norm': t_norm_operator,
        'category_mappings': category_mappings_copy,
        'categorical_conditions_decoded': categorical_conditions
    }
    
    return (
        variable_cutpoint_sets,
        linguistic_term_sets,
        fuzzy_rule_set,
        evaluation_summary
    )