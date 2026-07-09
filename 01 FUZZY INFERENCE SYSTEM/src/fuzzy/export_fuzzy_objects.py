# src/fuzzy/export_fuzzy_objects.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List


def plot_membership_functions(
    linguistic_terms: Dict[str, Dict],
    variables_to_plot: List[str] = None,
    figsize_per_var: tuple = (10, 4),
    save_path: str = None
):
    """Plot membership functions for fuzzy variables.

    This function generates plots of trapezoidal membership functions for specified
    fuzzy variables. Each variable is displayed in a separate subplot showing all
    its linguistic terms with filled areas under the curves. The function supports
    selective variable plotting and optional figure saving.

    Args:
        linguistic_terms (Dict[str, Dict]): Dictionary containing linguistic terms
                                           in format {variable: {term: (a, b, c, d)}}.
        variables_to_plot (List[str]): List of variable names to plot. If None,
                                       all variables are plotted.
        figsize_per_var (tuple): Figure size per variable as (width, height).
                                Default is (10, 4).
        save_path (str): File path for saving the figure. If None, figure is not saved.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    
    if variables_to_plot is None:
        variables_to_plot = list(linguistic_terms.keys())
    
    n_vars = len(variables_to_plot)
    
    fig, axes = plt.subplots(
        n_vars, 1, 
        figsize=(figsize_per_var[0], figsize_per_var[1] * n_vars)
    )
    
    if n_vars == 1:
        axes = [axes]
    
    for idx, var_name in enumerate(variables_to_plot):
        
        if var_name not in linguistic_terms:
            print(f"[WARNING] Variable '{var_name}' not found in linguistic_terms")
            continue
        
        ax = axes[idx]
        terms = linguistic_terms[var_name]
        
        all_params = []
        for params in terms.values():
            if len(params) == 4:
                all_params.extend(params)
        
        if not all_params:
            continue
        
        x_min = min(all_params)
        x_max = max(all_params)
        x_range = x_max - x_min
        x_min -= x_range * 0.05
        x_max += x_range * 0.05
        
        x = np.linspace(x_min, x_max, 1000)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(terms)))
        
        for (term_name, params), color in zip(terms.items(), colors):
            if len(params) != 4:
                continue
            
            a, b, c, d = params
            
            y = np.zeros_like(x)
            
            for i, val in enumerate(x):
                if val <= a or val >= d:
                    y[i] = 0.0
                elif a < val < b:
                    y[i] = (val - a) / (b - a) if (b - a) > 0 else 0.0
                elif b <= val <= c:
                    y[i] = 1.0
                elif c < val < d:
                    y[i] = (d - val) / (d - c) if (d - c) > 0 else 0.0
            
            ax.plot(x, y, label=term_name, color=color, linewidth=2)
            ax.fill_between(x, 0, y, alpha=0.2, color=color)
        
        ax.set_xlabel('Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Membership Degree', fontsize=11, fontweight='bold')
        ax.set_title(f'Variable: {var_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Membership functions saved to: {save_path}")
    
    plt.show()
    
    return fig


def summarize_fuzzy_rules(
    fuzzy_rules: List[Dict],
    linguistic_terms: Dict[str, Dict],
    top_n: int = 10,
    sort_by: str = 'logit'
):
    """Generate summary of fuzzy rules for paper presentation.

    This function analyzes and presents a comprehensive summary of the fuzzy rule
    system including general statistics, variable usage, linguistic term distribution,
    and detailed information about top-ranked rules. The summary is formatted for
    academic paper presentation.

    Args:
        fuzzy_rules (List[Dict]): List of fuzzy rule dictionaries containing
                                  antecedents, tree_id, logit scores, and
                                  predicted classes.
        linguistic_terms (Dict[str, Dict]): Dictionary of linguistic terms per variable.
        top_n (int): Number of top rules to display in detail. Default is 10.
        sort_by (str): Sorting criterion, either 'logit' for importance-based
                      sorting or 'tree_id' for tree-based ordering. Default is 'logit'.

    Returns:
        pd.DataFrame: DataFrame containing summary information for top N rules
                     with columns for rank, tree, class, logit, antecedents count,
                     and rule description.
    """
    
    print(f"\n[INFO] -------------------- Fuzzy Rule System Summary --------------------")
    
    total_rules = len(fuzzy_rules)
    tree_ids = set(r.get('tree_id', r.get('tree_idx', -1)) for r in fuzzy_rules)
    n_trees = len(tree_ids)
    
    rules_c0 = [r for r in fuzzy_rules if r.get('predicted_class') == 0]
    rules_c1 = [r for r in fuzzy_rules if r.get('predicted_class') == 1]
    
    print(f"\n[INFO] General Statistics:")
    print(f"  Total Rules:     {total_rules}")
    print(f"  Total Trees:     {n_trees}")
    print(f"  Class 0 Rules:   {len(rules_c0)} ({len(rules_c0)/total_rules*100:.1f}%)")
    print(f"  Class 1 Rules:   {len(rules_c1)} ({len(rules_c1)/total_rules*100:.1f}%)")
    
    all_vars = set()
    for rule in fuzzy_rules:
        for ant in rule.get('antecedents', []):
            all_vars.add(ant.get('variable'))
    
    print(f"\n[INFO] Variables:")
    print(f"  Total Variables: {len(all_vars)}")
    print(f"  Variables: {', '.join(sorted(all_vars))}")
    
    total_terms = sum(len(terms) for terms in linguistic_terms.values())
    avg_terms = total_terms / len(linguistic_terms) if linguistic_terms else 0
    
    print(f"\n[INFO] Linguistic Terms:")
    print(f"  Total Terms:     {total_terms}")
    print(f"  Avg per Variable: {avg_terms:.1f}")
    
    print(f"\n[INFO] -------------------- Top {top_n} Rules (sorted by {sort_by}) --------------------")
    
    if sort_by == 'logit':
        sorted_rules = sorted(fuzzy_rules, key=lambda r: abs(r.get('logit', r.get('logit_score', 0))), reverse=True)
    else:
        sorted_rules = fuzzy_rules
    
    rule_data = []
    
    for i, rule in enumerate(sorted_rules[:top_n], 1):
        
        tree_id = rule.get('tree_id', rule.get('tree_idx', -1))
        logit = rule.get('logit', rule.get('logit_score', 0))
        predicted_class = rule.get('predicted_class', -1)
        
        antecedents = []
        for ant in rule.get('antecedents', []):
            var = ant.get('variable', 'unknown')
            terms = ant.get('linguistic_terms', [])
            
            if len(terms) == 1:
                antecedents.append(f"{var} IS {terms[0]}")
            else:
                antecedents.append(f"{var} IS ({' OR '.join(terms)})")
        
        rule_str = " AND ".join(antecedents)
        
        rule_data.append({
            'Rank': i,
            'Tree': tree_id,
            'Class': predicted_class,
            'Logit': f"{logit:+.3f}",
            'Antecedents': len(antecedents),
            'Rule': rule_str[:80] + '...' if len(rule_str) > 80 else rule_str
        })
    
    df = pd.DataFrame(rule_data)
    print(f"\n{df.to_string(index=False)}")
    
    return df

    
def export_rules_for_paper(
    fuzzy_rules: List[Dict],
    linguistic_terms: Dict[str, Dict],
    output_file: str = 'fuzzy_rules_paper.txt',
    max_rules: int = 20
):
    """
    Export fuzzy rules in LaTeX-compatible text format for academic publication.
    
    Generates a comprehensive text file containing system overview, linguistic terms,
    top rules by importance, and LaTeX table format examples suitable for inclusion
    in research papers.
    
    Args:
        fuzzy_rules: List of fuzzy rule dictionaries containing antecedents, logit
            contributions, predicted classes, and tree identifiers.
        linguistic_terms: Dictionary mapping variable names to their linguistic term
            definitions with trapezoidal membership function parameters.
        output_file: Path to output text file for exported rules.
        max_rules: Maximum number of top rules to export based on absolute logit
            contribution magnitude.
    
    Returns:
        None. Writes formatted rule system description to specified output file.
    """
    
    sorted_rules = sorted(
        fuzzy_rules, 
        key=lambda r: abs(r.get('logit', r.get('logit_score', 0))), 
        reverse=True
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write("\n[INFO] " + "-"*88 + "\n")
        f.write("FUZZY RULE SYSTEM - FORMATTED FOR PAPER\n")
        f.write("-"*100 + "\n\n")
        
        f.write("System Overview:\n")
        f.write(f"  Total Rules: {len(fuzzy_rules)}\n")
        f.write(f"  Variables: {len(linguistic_terms)}\n")
        f.write(f"  Trees: {len(set(r.get('tree_id', -1) for r in fuzzy_rules))}\n")
        f.write("\n\n")
        
        f.write("Variables and Linguistic Terms:\n")
        f.write("-"*100 + "\n")
        for var_name, terms in sorted(linguistic_terms.items()):
            f.write(f"\n{var_name}:\n")
            for term_name, params in terms.items():
                a, b, c, d = params
                f.write(f"  - {term_name}: [{a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}]\n")
        
        f.write("\n\n")
        
        f.write(f"Top {max_rules} Rules (by importance):\n")
        f.write("-"*100 + "\n\n")
        
        for i, rule in enumerate(sorted_rules[:max_rules], 1):
            
            tree_id = rule.get('tree_id', rule.get('tree_idx', -1))
            logit = rule.get('logit', rule.get('logit_score', 0))
            predicted_class = rule.get('predicted_class', -1)
            
            f.write(f"Rule {i} (Tree {tree_id}, Class {predicted_class}, Logit {logit:+.3f}):\n")
            f.write("-"*100 + "\n")
            
            f.write("IF ")
            antecedents = []
            for j, ant in enumerate(rule.get('antecedents', [])):
                var = ant.get('variable', 'unknown')
                terms = ant.get('linguistic_terms', [])
                
                if len(terms) == 1:
                    antecedents.append(f"{var} IS {terms[0]}")
                else:
                    antecedents.append(f"{var} IS ({' OR '.join(terms)})")
            
            f.write("\n   AND ".join(antecedents))
            
            class_label = "Good" if predicted_class == 0 else "Bad"
            f.write(f"\nTHEN Prediction = {class_label}\n")
            f.write(f"     (Logit contribution: {logit:+.3f})\n")
            f.write("\n")
        
        f.write("\n\n")
        f.write("-"*100 + "\n")
        f.write("LATEX FORMAT (example for 3 rules)\n")
        f.write("-"*100 + "\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sample Fuzzy Rules}\n")
        f.write("\\begin{tabular}{|c|l|c|c|}\n")
        f.write("\\hline\n")
        f.write("Rule & Antecedents & Class & Logit \\\\\n")
        f.write("\\hline\n")
        
        for i, rule in enumerate(sorted_rules[:3], 1):
            tree_id = rule.get('tree_id', -1)
            logit = rule.get('logit', rule.get('logit_score', 0))
            predicted_class = rule.get('predicted_class', -1)
            
            ants = rule.get('antecedents', [])
            if len(ants) <= 2:
                ant_str = " AND ".join([f"{a.get('variable')} IS {a.get('linguistic_terms', [''])[0]}" 
                                       for a in ants[:2]])
            else:
                ant_str = f"{len(ants)} conditions"
            
            class_label = "Good" if predicted_class == 0 else "Bad"
            
            f.write(f"{i} & {ant_str} & {class_label} & {logit:+.3f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n[INFO] Rules exported to: {output_file}")

def plot_membership_functions_paper(
    linguistic_terms: Dict[str, Dict],
    variables: list = None,
    output_path: str = 'membership_functions_paper.png'
):
    """Generate compact 2x2 figure with key variables for paper.

    This function creates a publication-ready figure displaying membership functions
    for four key variables in a 2x2 grid layout. The figure uses consistent white
    background, bold labels, and optimized formatting for academic paper inclusion.

    Args:
        linguistic_terms (Dict[str, Dict]): Dictionary of linguistic terms with
                                           trapezoid parameters.
        variables (list): List of variable names to plot. Must contain exactly
                         4 variables for 2x2 layout.
        output_path (str): Output file path for PNG image. Default is
                          'membership_functions_paper.png'.

    Returns:
        str: Path to the saved figure file.
    """
    
    variables = variables
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), facecolor='white')
    axes = axes.flatten()
    
    for idx, var_name in enumerate(variables):
        ax = axes[idx]
        
        ax.set_facecolor('white')
        
        if var_name not in linguistic_terms:
            print(f"[WARNING] '{var_name}' not in linguistic_terms")
            ax.text(0.5, 0.5, f'{var_name}\nnot available', 
                   ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(var_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel('μ', fontsize=10, fontweight='bold')
            continue
        
        terms = linguistic_terms[var_name]
        
        if not terms:
            print(f"[WARNING] No terms for '{var_name}'")
            continue
        
        all_params = []
        for params in terms.values():
            if len(params) == 4:
                all_params.extend(params)
        
        if not all_params:
            continue
        
        x_min = min(all_params)
        x_max = max(all_params)
        x_range = x_max - x_min
        x_min -= x_range * 0.02
        x_max += x_range * 0.02
        
        x = np.linspace(x_min, x_max, 500)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(terms)))
        
        for (term_name, params), color in zip(terms.items(), colors):
            if len(params) != 4:
                continue
            
            a, b, c, d = params
            y = np.zeros_like(x)
            
            mask1 = (x > a) & (x < b)
            mask2 = (x >= b) & (x <= c)
            mask3 = (x > c) & (x < d)
            
            y[mask1] = (x[mask1] - a) / (b - a) if (b - a) > 0 else 0.0
            y[mask2] = 1.0
            y[mask3] = (d - x[mask3]) / (d - c) if (d - c) > 0 else 0.0
            
            ax.plot(x, y, linewidth=2, label=term_name, color=color)
            ax.fill_between(x, 0, y, alpha=0.2, color=color)
        
        ax.set_xlabel(var_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylabel('μ', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.95, 
                 edgecolor='gray', fancybox=False)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"[INFO] Paper membership functions saved: {output_path}")
    print(f"[INFO] Variables plotted: {', '.join(variables)}")
    
    return output_path


