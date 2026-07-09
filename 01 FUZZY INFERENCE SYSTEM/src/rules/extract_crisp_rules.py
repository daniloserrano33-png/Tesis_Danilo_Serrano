# src/rules/extract_crisp_rules.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score
import catboost
import json
import tempfile
import os
from joblib import Parallel, delayed
import time


def _serialize_model_to_json(classifier: catboost.CatBoostClassifier) -> Dict[str, Any]:
    """Extract internal model structure as JSON dictionary.

    This function exports the CatBoost model architecture, including split points
    and leaf values, to a JSON format using a temporary file. The JSON structure
    contains the complete tree ensemble representation.

    Args:
        classifier (catboost.CatBoostClassifier): Trained CatBoost classification model.

    Returns:
        Dict[str, Any]: Parsed JSON structure containing the tree ensemble.

    Raises:
        RuntimeError: If model serialization fails during the export process.
    """
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(
            mode='w+', 
            suffix='.json', 
            delete=False
        ) as temp_handle:
            temp_file_path = temp_handle.name
        
        classifier.save_model(temp_file_path, format='json')
        
        with open(temp_file_path, 'r', encoding='utf-8') as json_handle:
            model_structure = json.load(json_handle)
            
        return model_structure
        
    except Exception as error:
        raise RuntimeError(
            f"Model serialization failed: {str(error)}"
        )
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _decompose_tree_architecture(
    tree_dictionary: Dict[str, Any], 
    feature_labels: List[str]
) -> Tuple[List[Dict], List[float]]:
    """Parse oblivious tree structure into split metadata and leaf values.

    This function processes a single tree from the CatBoost model JSON structure,
    extracting split definitions and terminal node scores. The function operates
    on pre-encoded categorical features, treating all features as numeric. Split
    metadata includes feature identifiers, split types, and threshold values.

    Args:
        tree_dictionary (Dict[str, Any]): Single tree structure from model JSON.
        feature_labels (List[str]): Feature name lookup table.

    Returns:
        Tuple[List[Dict], List[float]]: A tuple containing:
            - List of split metadata dictionaries with feature, type, and threshold
            - List of logit scores for terminal nodes
    """
    split_specifications = tree_dictionary.get('splits', [])
    terminal_node_scores = tree_dictionary.get('leaf_values', [])
    
    parsed_splits = []
    
    for i, split_definition in enumerate(split_specifications):
        feature_index = None
        
        if 'float_feature_index' in split_definition:
            feature_index = split_definition['float_feature_index']
        
        if feature_index is None or feature_index >= len(feature_labels):
            parsed_splits.append({
                'feature': f'unknown_feature_{i}',
                'type': 'numeric',
                'threshold': split_definition.get('border', 0.0),
                'is_unknown': True
            })
            continue
        
        feature_identifier = feature_labels[feature_index]
        threshold = split_definition.get('border', 0.0)
        
        parsed_splits.append({
            'feature': feature_identifier,
            'type': 'numeric',
            'threshold': threshold
        })
    
    return parsed_splits, terminal_node_scores


def _reconstruct_decision_paths(
    split_metadata: List[Dict], 
    leaf_scores: List[float]
) -> List[Dict]:
    """Generate explicit decision paths for all leaf nodes in an oblivious tree.

    This function reconstructs the decision path to each terminal node by decoding
    the binary representation of leaf indices. In oblivious trees, the tree depth
    equals the number of splits, and the total number of leaves equals 2 raised to
    the power of the tree depth. Each leaf index encodes its path as a binary number,
    where bit i indicates whether the path went right (1) or left (0) at level i.

    Args:
        split_metadata (List[Dict]): Ordered list of split definitions.
        leaf_scores (List[float]): Logit value for each terminal node.

    Returns:
        List[Dict]: List of path dictionaries, each containing:
            - leaf_index: Integer identifier for the leaf node
            - conditions: List of human-readable decision conditions
            - logit_contribution: Float representing the leaf's logit score
    """
    tree_depth = len(split_metadata)
    total_leaves = 2 ** tree_depth
    
    valid_leaf_count = min(total_leaves, len(leaf_scores))
    
    decision_paths = []
    
    for leaf_number in range(valid_leaf_count):
        path_conditions = []
        
        for depth_level in range(tree_depth):
            split_info = split_metadata[depth_level]
            
            bit_position = tree_depth - 1 - depth_level
            took_right_branch = bool((leaf_number >> bit_position) & 1)
            
            threshold = split_info['threshold']
            feature = split_info['feature']
            
            if took_right_branch:
                condition = f"{feature} > {threshold:.6f}"
            else:
                condition = f"{feature} <= {threshold:.6f}"
            
            path_conditions.append(condition)
        
        decision_paths.append({
            'leaf_index': leaf_number,
            'conditions': path_conditions,
            'logit_contribution': float(
                leaf_scores[leaf_number] 
                if leaf_number < len(leaf_scores) 
                else 0.0
            )
        })
    
    return decision_paths


def extract_rules_from_trees(
    classifier: catboost.CatBoostClassifier,
    feature_matrix: pd.DataFrame,
    target_labels: pd.Series,
    tree_index_list: Optional[List[int]] = None,
    min_confidence: float = 0.0,
    min_support: float = 0.0,
    parallel_jobs: int = -1
) -> List[Dict]:
    """Extract decision rules from specified CatBoost trees.

    This function extracts interpretable decision rules from a trained CatBoost model
    by traversing tree structures and computing coverage statistics. The function
    operates on pre-encoded categorical features, treating all features as numeric.
    Rules contain logit scores for additive inference and precision-based quality
    metrics. The final prediction is obtained by summing logits and applying the
    sigmoid transformation.

    Args:
        classifier (catboost.CatBoostClassifier): Trained CatBoost model.
        feature_matrix (pd.DataFrame): Training features with encoded categorical variables.
        target_labels (pd.Series): Training labels.
        tree_index_list (Optional[List[int]]): Specific trees to extract. None extracts all trees.
        min_confidence (float): Minimum absolute logit threshold for rule inclusion.
        min_support (float): Minimum coverage fraction threshold for rule inclusion.
        parallel_jobs (int): CPU cores for parallel extraction.

    Returns:
        List[Dict]: List of rule dictionaries, each containing:
            - tree_idx: Tree index in the ensemble
            - leaf_idx: Leaf node identifier
            - rule_conditions: List of decision conditions
            - logit_score: Continuous logit contribution
            - predicted_class: Binary class prediction (0 or 1)
            - confidence: Absolute value of logit score
            - support: Fraction of samples covered by the rule
            - precision: Proportion of correct predictions in coverage region
            - purity: Alternative metric equivalent to precision
            - quality_score: Overall quality metric based on precision
            - samples_covered: Number of samples matching the rule
            - class_0_count: Count of class 0 samples in coverage
            - class_1_count: Count of class 1 samples in coverage
            - features_involved: Sorted list of features used in conditions
            - feature_count: Number of distinct features
            - condition_count: Number of decision conditions
            - tp: True positives in coverage region
            - fp: False positives in coverage region
    """
    print("\n[INFO] -------------------- CatBoost Rule Extraction --------------------")
    
    model_json = _serialize_model_to_json(classifier)
    tree_ensemble = model_json.get('oblivious_trees', [])
    feature_names = list(feature_matrix.columns)
    
    y_array = (
        target_labels.values 
        if isinstance(target_labels, pd.Series) 
        else np.array(target_labels)
    )
    
    total_class_0 = np.sum(y_array == 0)
    total_class_1 = np.sum(y_array == 1)
    total_samples = len(y_array)
    
    print(f"\n[INFO] Dataset statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Class 0: {total_class_0:,} ({total_class_0/total_samples*100:.1f}%)")
    print(f"  Class 1: {total_class_1:,} ({total_class_1/total_samples*100:.1f}%)")
    
    print(f"\n[INFO] Creating CatBoost Pool...")
    print(f"   All features treated as numeric (pre-encoded)")
    
    pool = catboost.Pool(feature_matrix)
    
    if tree_index_list is None:
        tree_index_list = list(range(len(tree_ensemble)))
    
    print(f"\n[INFO] Processing {len(tree_index_list)} trees...")
    
    all_rules = []
    
    for tree_idx in tree_index_list:
        tree = tree_ensemble[tree_idx]
        leaf_values = tree.get('leaf_values', [])
        
        split_info, _ = _decompose_tree_architecture(tree, feature_names)
        paths = _reconstruct_decision_paths(split_info, leaf_values)
        
        try:
            leaf_indices_2d = classifier.calc_leaf_indexes(pool, tree_idx, tree_idx + 1)
            leaf_indices = leaf_indices_2d[:, 0]
        except Exception as e:
            print(f"  [WARNING] Tree {tree_idx}: {e}")
            continue
        
        unique_leaves, counts = np.unique(leaf_indices, return_counts=True)
        
        for leaf_idx, count in zip(unique_leaves, counts):
            if leaf_idx >= len(leaf_values):
                continue
            
            logit_value = leaf_values[leaf_idx]
            
            if abs(logit_value) < 1e-10:
                continue
            
            matching_path = None
            for path in paths:
                if path['leaf_index'] == leaf_idx:
                    matching_path = path
                    break
            
            if matching_path is None:
                continue
            
            coverage_mask = (leaf_indices == leaf_idx)
            covered_labels = y_array[coverage_mask]
            coverage_fraction = count / len(feature_matrix)
            
            if coverage_fraction < min_support:
                continue
                
            if abs(logit_value) < min_confidence:
                continue
            
            predicted_class = 1 if logit_value > 0 else 0
            
            if predicted_class == 1:
                tp = np.sum(covered_labels == 1)
                fp = np.sum(covered_labels == 0)
            else:
                tp = np.sum(covered_labels == 0)
                fp = np.sum(covered_labels == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            purity = precision
            
            quality_score = precision
            
            involved_features = set()
            for cond in matching_path['conditions']:
                for fname in feature_names:
                    if fname in cond:
                        involved_features.add(fname)
            
            rule = {
                'tree_idx': tree_idx,
                'leaf_idx': int(leaf_idx),
                'rule_conditions': matching_path['conditions'],
                'logit_score': logit_value,
                'predicted_class': predicted_class,
                'confidence': round(abs(logit_value), 6),
                'support': round(coverage_fraction, 6),
                'precision': round(precision, 4),
                'purity': round(purity, 4),
                'quality_score': round(quality_score, 4),
                'samples_covered': int(count),
                'class_0_count': int(np.sum(covered_labels == 0)),
                'class_1_count': int(np.sum(covered_labels == 1)),
                'features_involved': sorted(list(involved_features)),
                'feature_count': len(involved_features),
                'condition_count': len(matching_path['conditions']),
                'tp': int(tp),
                'fp': int(fp)
            }
            
            all_rules.append(rule)
    
    rules_c0 = [r for r in all_rules if r['predicted_class'] == 0]
    rules_c1 = [r for r in all_rules if r['predicted_class'] == 1]
    
    if rules_c0:
        prec_c0 = [r['precision'] for r in rules_c0]
        avg_prec_c0 = np.mean(prec_c0)
        high_quality_c0 = sum(1 for p in prec_c0 if p > 0.7)
    else:
        avg_prec_c0 = 0
        high_quality_c0 = 0
    
    if rules_c1:
        prec_c1 = [r['precision'] for r in rules_c1]
        avg_prec_c1 = np.mean(prec_c1)
        high_quality_c1 = sum(1 for p in prec_c1 if p > 0.7)
    else:
        avg_prec_c1 = 0
        high_quality_c1 = 0
    
    print(f"\n   [INFO] Extracted {len(all_rules):,} rules")
    print(f"   [INFO] All conditions are numeric (categorical features encoded)")
    
    print(f"\n   [INFO] Metrics validation:")
    print(f"     Class 0: {len(rules_c0):,} rules")
    print(f"       - Avg Precision: {avg_prec_c0:.3f}")
    print(f"       - High quality (>0.7): {high_quality_c0} ({high_quality_c0/len(rules_c0)*100 if rules_c0 else 0:.1f}%)")
    
    print(f"     Class 1: {len(rules_c1):,} rules")
    print(f"       - Avg Precision: {avg_prec_c1:.3f}")
    print(f"       - High quality (>0.7): {high_quality_c1} ({high_quality_c1/len(rules_c1)*100 if rules_c1 else 0:.1f}%)")
    
    all_precisions = [r['precision'] for r in all_rules]
    if all_precisions:
        print(f"\n   [INFO] Quality distribution:")
        print(f"     Min precision:    {np.min(all_precisions):.3f}")
        print(f"     25th percentile:  {np.percentile(all_precisions, 25):.3f}")
        print(f"     Median:           {np.median(all_precisions):.3f}")
        print(f"     75th percentile:  {np.percentile(all_precisions, 75):.3f}")
        print(f"     Max precision:    {np.max(all_precisions):.3f}")
    
    print("\n[INFO] -------------------- Extraction Complete --------------------")
    
    return all_rules


def get_catboost_trees(
    model: catboost.CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_trees: Optional[int] = None,
    max_samples: int = 50000,
    parallel_jobs: int = -1
) -> List[int]:
    """Select balanced trees from CatBoost ensemble.

    This function evaluates all trees in the ensemble and selects a subset that
    maintains balance between positive and negative predictions. If n_trees is
    None or exceeds the total number of trees, all trees are returned. The
    selection process uses a greedy algorithm that prioritizes high-quality trees
    while constraining the cumulative bias to prevent class imbalance.

    Args:
        model (catboost.CatBoostClassifier): Trained CatBoost model.
        X (pd.DataFrame): Feature matrix for tree evaluation.
        y (pd.Series): Target labels for tree evaluation.
        n_trees (Optional[int]): Number of trees to select. None or values greater
                                  than total trees returns all trees.
        max_samples (int): Maximum number of samples for evaluation efficiency.
        parallel_jobs (int): Number of CPU cores for parallel evaluation.

    Returns:
        List[int]: List of selected tree indices maintaining balance constraints.
    """
    print("\n[INFO] -------------------- Tree Selection Process --------------------")
    
    num_trees = model.tree_count_
    
    if n_trees is None or n_trees >= num_trees:
        print(f"[INFO] Using ALL {num_trees} trees (no selection needed)")
        print("\n[INFO] -------------------- Selection Complete --------------------")
        return list(range(num_trees))
    
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    
    if len(X) > max_samples:
        print(f"[INFO] Sampling {max_samples:,} of {len(X):,} instances...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=max_samples, random_state=42)
        indices = next(sss.split(X, y_array))[1]
        X_eval = X.iloc[indices]
        y_eval = y_array[indices]
    else:
        X_eval = X
        y_eval = y_array
    
    pool = catboost.Pool(X_eval)
    
    print(f"[INFO] Evaluating {num_trees} trees...")
    print(f"[INFO] All features treated as numeric (pre-encoded)")
    
    start_time = time.time()
    
    def evaluate_tree_balance(args):
        tree_idx, model, pool, labels = args
        
        try:
            leaf_indices_2d = model.calc_leaf_indexes(pool, tree_idx, tree_idx + 1)
            leaf_indices = leaf_indices_2d[:, 0]
            
            model_json = _serialize_model_to_json(model)
            tree = model_json['oblivious_trees'][tree_idx]
            leaf_values = tree.get('leaf_values', [])
            
            logits = np.array([leaf_values[idx] for idx in leaf_indices])
            tree_bias = np.mean(logits)
            
            predictions = (logits > 0).astype(int)
            
            recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
            precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
            f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
            
            quality = 0.5 * recall + 0.3 * f1 + 0.2 * precision
            
            return (tree_idx, quality, tree_bias, f1)
            
        except Exception as e:
            return (tree_idx, 0.0, 0.0, 0.0)
    
    eval_tasks = [(i, model, pool, y_eval) for i in range(num_trees)]
    
    results = Parallel(n_jobs=parallel_jobs, verbose=0, backend='threading')(
        delayed(evaluate_tree_balance)(task) for task in eval_tasks
    )
    
    tree_quality = {}
    tree_bias = {}
    tree_f1 = {}
    
    for tree_idx, quality, bias, f1 in results:
        tree_quality[tree_idx] = quality
        tree_bias[tree_idx] = bias
        tree_f1[tree_idx] = f1
    
    selected_trees = []
    cumulative_bias = 0.0
    
    candidate_trees = sorted(
        tree_quality.keys(),
        key=lambda idx: tree_quality[idx],
        reverse=True
    )
    
    print(f"\n[INFO] Selecting {n_trees} trees with balance constraint...")
    
    for tree_idx in candidate_trees:
        if len(selected_trees) >= n_trees:
            break
        
        new_cumulative_bias = cumulative_bias + tree_bias[tree_idx]
        
        if len(selected_trees) < 20:
            selected_trees.append(tree_idx)
            cumulative_bias = new_cumulative_bias
        else:
            avg_bias_with_tree = new_cumulative_bias / (len(selected_trees) + 1)
            
            if abs(avg_bias_with_tree) < 0.5:
                selected_trees.append(tree_idx)
                cumulative_bias = new_cumulative_bias
    
    selected_biases = [tree_bias[idx] for idx in selected_trees]
    selected_qualities = [tree_quality[idx] for idx in selected_trees]
    
    final_avg_bias = np.mean(selected_biases)
    final_avg_quality = np.mean(selected_qualities)
    
    n_positive = sum(1 for b in selected_biases if b > 0.1)
    n_negative = sum(1 for b in selected_biases if b < -0.1)
    n_neutral = len(selected_biases) - n_positive - n_negative
    
    eval_time = time.time() - start_time
    
    print(f"\n  [INFO] Selection completed in {eval_time:.1f}s")
    print(f"  [INFO] Selected: {len(selected_trees)} trees")
    
    print(f"\n  [INFO] Tree Bias Distribution:")
    print(f"    Positive bias (Class 1): {n_positive} trees")
    print(f"    Negative bias (Class 0): {n_negative} trees")
    print(f"    Neutral:                 {n_neutral} trees")
    
    print(f"\n  [INFO] Final Statistics:")
    print(f"    Average bias: {final_avg_bias:+.4f}")
    print(f"    Average quality: {final_avg_quality:.4f}")
    print(f"    Bias range: [{min(selected_biases):+.3f}, {max(selected_biases):+.3f}]")
    
    if abs(final_avg_bias) < 0.3:
        print(f"\n  [INFO] Balance achieved")
    else:
        print(f"\n  [WARNING] Balance suboptimal: |avg_bias| = {abs(final_avg_bias):.3f}")
    
    print("\n[INFO] -------------------- Selection Complete --------------------")
    
    return selected_trees


def remove_duplicate_rules(rules: List[Dict]) -> List[Dict]:
    """
    Removes duplicate rules, keeping only unique ones.
    When duplicates exist, keeps the one with highest confidence.
    
    Args:
        rules: List of rules to deduplicate
    
    Returns:
        List of unique rules
    """
    if not rules:
        return []
    
    # Create a dictionary to store unique rules
    # Key: frozenset of sorted conditions
    # Value: best rule for that condition set
    unique_rules = {}
    
    for rule in rules:
        # Create a unique key from the conditions
        conditions_sorted = tuple(sorted(rule['rule_conditions']))
        print(f"\n[INFO] Processing rule with conditions: {conditions_sorted}")
        # If this condition set hasn't been seen, or if this rule is better
        if conditions_sorted not in unique_rules:
            unique_rules[conditions_sorted] = rule
        else:
            # Keep the rule with higher confidence (or you can use lift, support, etc.)
            if rule['confidence'] > unique_rules[conditions_sorted]['confidence']:
                unique_rules[conditions_sorted] = rule
    
    # Convert back to list
    unique_rules_list = list(unique_rules.values())
    
    print(f"\n[INFO] Original rules: {len(rules)}")
    print(f"\n[INFO] Unique rules: {len(unique_rules_list)}")
    print(f"\n[INFO] Duplicates removed: {len(rules) - len(unique_rules_list)}")
    
    return unique_rules_list


def convert_rules_to_simple_format(rules: List[Dict]) -> List[Dict]:
    """
    Converts rules to simple format: tree_id, conditions, prediction.
    
    Args:
        rules: List of rules from BJA_extract_rules_from_tree
    
    Returns:
        List of rules in simple format
    """
    simple_rules = []
    
    for rule in rules:
        simple_rule = {
            'tree_id': rule['tree_idx'],
            'conditions': rule['rule_conditions'],
            'predicted_class': rule['predicted_class'],
            'logit_score': rule['logit_score']
            
            ,
            'confidence': rule['confidence'],
            'support': rule['support'],
            'precision': rule['precision']
        }
        simple_rules.append(simple_rule)
    
    return simple_rules


def rank_trees(all_rules):
    """Rank trees by quality score based on precision and rule count.

    This function evaluates each tree in the rule set by computing a composite
    score that combines average precision and the number of rules generated by
    the tree. The score is calculated as the product of average precision and
    rule count. Trees are ranked in descending order by this score.

    Args:
        all_rules (list): List of rule dictionaries, each containing a 'tree_id'
                         key and optionally a 'precision' key.

    Returns:
        list: List of dictionaries sorted by score in descending order. Each
              dictionary contains:
                - tree_id (int): Tree identifier
                - score (float): Quality score (average precision × rule count)
                - n_rules (int): Number of rules generated by the tree
    """
    
    tree_ids = sorted(set(r['tree_id'] for r in all_rules))
    
    tree_scores = []
    for tree_id in tree_ids:
        tree_rules = [r for r in all_rules if r['tree_id'] == tree_id]
        
        if not tree_rules:
            continue
        
        avg_precision = np.mean([r.get('precision', 0) for r in tree_rules])
        n_rules = len(tree_rules)
        score = avg_precision * n_rules
        
        tree_scores.append({
            'tree_id': tree_id,
            'score': score,
            'n_rules': n_rules
        })
    
    tree_scores = sorted(tree_scores, key=lambda x: x['score'], reverse=True)
    
    return tree_scores


def compare_tree_strategies(all_rules, n_trees_options=[15, 20, 25, 30]):
    """Compare different tree count strategies and display balance metrics.

    This function evaluates multiple tree selection strategies by ranking trees
    using a shared scoring function, extracting rules for each specified tree count,
    analyzing class balance through logit score summation, and identifying the
    configuration with the best balance (net sum closest to zero).

    Args:
        all_rules (list): Complete list of decision rules with tree_id, predicted_class,
                         and logit_score attributes.
        n_trees_options (list): List of integers representing different tree counts
                               to evaluate. Default is [15, 20, 25, 30].

    Returns:
        pd.DataFrame: DataFrame containing evaluation results with columns for
                     n_trees, n_rules, class-specific rule counts, logit sums,
                     net sum, and absolute net sum.
    """
    
    print("\n[INFO] -------------------- Comparing Tree Strategies --------------------")
    
    tree_scores = rank_trees(all_rules)
    
    print(f"[INFO] Total trees available: {len(tree_scores)}")
    
    results = []
    
    for n_trees in n_trees_options:
        
        selected_tree_ids = [t['tree_id'] for t in tree_scores[:n_trees]]
        
        selected_rules = [r for r in all_rules if r['tree_id'] in selected_tree_ids]
        
        rules_c0 = [r for r in selected_rules if r['predicted_class'] == 0]
        rules_c1 = [r for r in selected_rules if r['predicted_class'] == 1]
        
        sum_c0 = np.sum([r['logit_score'] for r in rules_c0])
        sum_c1 = np.sum([r['logit_score'] for r in rules_c1])
        net_sum = sum_c0 + sum_c1
        
        results.append({
            'n_trees': n_trees,
            'n_rules': len(selected_rules),
            'n_rules_c0': len(rules_c0),
            'n_rules_c1': len(rules_c1),
            'sum_c0': sum_c0,
            'sum_c1': sum_c1,
            'net_sum': net_sum,
            'abs_net': abs(net_sum)
        })
        
        print(f"\n[INFO] {n_trees} trees:")
        print(f"  Rules: {len(selected_rules)} (C0:{len(rules_c0)}, C1:{len(rules_c1)})")
        print(f"  Sum C0: {sum_c0:.1f}")
        print(f"  Sum C1: {sum_c1:.1f}")
        print(f"  Net:    {net_sum:+.1f}")
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['abs_net'].idxmin()
    best = results_df.loc[best_idx]
    
    print(f"\n[INFO] ----------------------------------------------------------------")
    print(f"[INFO] BEST BALANCE: {int(best['n_trees'])} trees")
    print(f"  Net sum: {best['net_sum']:+.1f} (closest to 0)")
    print(f"  Rules: {int(best['n_rules'])}")
    print(f"[INFO] ----------------------------------------------------------------")
    
    return results_df


def select_best_trees(all_rules, n_trees=20):
    """Select the best N complete trees based on quality score.

    This function ranks all available trees using a shared scoring function,
    selects the top N trees by score, extracts all rules from selected trees,
    and analyzes class balance through logit score summation for both classes.

    Args:
        all_rules (list): Complete list of decision rules with tree_id, predicted_class,
                         and logit_score attributes.
        n_trees (int): Number of top-ranked trees to select. Default is 20.

    Returns:
        list: List of selected rules from the top N trees, preserving all rule
              attributes including tree_id, predicted_class, and logit_score.
    """
    
    print(f"\n[INFO] -------------------- Selecting {n_trees} Best Trees --------------------")
    
    tree_scores = rank_trees(all_rules)
    
    selected_tree_ids = [t['tree_id'] for t in tree_scores[:n_trees]]
    
    selected_rules = [r for r in all_rules if r['tree_id'] in selected_tree_ids]
    
    rules_c0 = [r for r in selected_rules if r['predicted_class'] == 0]
    rules_c1 = [r for r in selected_rules if r['predicted_class'] == 1]
    
    sum_c0 = np.sum([r['logit_score'] for r in rules_c0])
    sum_c1 = np.sum([r['logit_score'] for r in rules_c1])
    net_sum = sum_c0 + sum_c1
    
    print(f"\n[INFO] Selected {len(selected_rules)} rules from {n_trees} trees")
    print(f"  Class 0: {len(rules_c0)} rules, sum={sum_c0:.1f}")
    print(f"  Class 1: {len(rules_c1)} rules, sum={sum_c1:.1f}")
    print(f"  Net sum: {net_sum:+.1f}")
    
    if abs(net_sum) < 50:
        print(f"  Good balance")
    else:
        print(f"  Imbalanced")
    
    print(f"[INFO] ----------------------------------------------------------------")
    
    return selected_rules