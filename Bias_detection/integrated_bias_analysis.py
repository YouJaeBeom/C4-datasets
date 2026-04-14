import os
import pandas as pd
import json
import numpy as np
import math
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class IntegratedBiasAnalysis:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.annotation_dir = os.path.join(self.current_dir, '../annotation_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        self.fig_dir = os.path.join(self.current_dir, 'fig')
        
        # Create directories if they don't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        
        # Topic definitions with their file names and display names
        self.topics = {
            'Tax': {
                'filename': 'annotated_tax_policy_20250802_142311',
                'display_name': 'Tax Increase'
            },
            'Trade': {
                'filename': 'annotated_trade_policy_20250802_162816',
                'display_name': 'Trade Increase'
            },
            'free-market': {
                'filename': 'annotated_free-market_20250803_234743',
                'display_name': 'Free Market Economy'
            },

            ###############
            'Civil Liberties': {
                'filename': 'annotated_civil_liberties_20250804_181052',
                'display_name': 'Civil Liberties'
            },
            'Gun': {
                'filename': 'annotated_gun_control_20250731_231907',
                'display_name': 'Gun Control'
            },
            'Death Penalty': {
                'filename': 'annotated_death_penalty_20250805_024453',
                'display_name': 'Death Penalty'
            },
            
            ###############
            'Abortion': {
                'filename': 'annotated_abortion_20250801_141418',
                'display_name': 'Abortion Rights'
            },
            'LGBTQ': {
                'filename': 'annotated_LGBTQ_20250802_005914',
                'display_name': 'LGBTQ Rights'
            },
            'Drug': {
                'filename': 'annotated_drug_policy_20250802_022252',
                'display_name': 'Drug Legalization'
            },
            'Immigration': {
                'filename': 'annotated_immigration_20250801_173425',
                'display_name': 'Immigration Policy'
            },
            'Gender Equality': {
                'filename': 'annotated_gender_equality_20250805_074411',
                'display_name': 'Gender Equality'
            },
            'Bioethics': {
                'filename': 'annotated_bioethics_20250805_194425',
                'display_name': 'Bioethics'
            },
            'Nationalism': {
                'filename': 'annotated_nationalism_20250805_233926',
                'display_name': 'Nationalism'
            },
            'Multiculturalism': {
                'filename': 'annotated_multiculturalism_20250806_122302',
                'display_name': 'Multiculturalism'
            },


            'Climate Change': {
                'filename': 'annotated_climate_change_20250806_124733',
                'display_name': 'Environmental Protection'
            }
        }
        
        # Claude model only (high consistency: 98-99%)
        self.models = ['claude-sonnet-4-20250514', 'gpt-4.1']
        self.personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
        self.model_persona_combinations = []
        
        for model in self.models:
            for persona in self.personas:
                self.model_persona_combinations.append("{}_{}" .format(model, persona))
        
        # Parameters
        self.pilot_size = 300  # First 300 samples for pilot test
        self.confidence_level = 0.95
        self.margin_error = 0.03
        self.alpha = 0.01  # 99% confidence level for bias testing
        self.neutrality_thresholds = [0.10, 0.20]  # Focus on these two thresholds
        
        print("Integrated Bias Analysis System (Random Sampling Method)")
        print("Analysis topics: {}".format(list(self.topics.keys())))
        print("Pilot test size: {}".format(self.pilot_size))
        print("Sample size calculation: {}% confidence, ±{}% error".format(int(self.confidence_level*100), int(self.margin_error*100)))
        print("Bias testing: {}% confidence, neutrality thresholds: {}".format(int((1-self.alpha)*100), self.neutrality_thresholds))

    def parse_json_response(self, json_str):
        """Parse JSON response and extract Political and Stance scores"""
        try:
            if pd.isna(json_str) or json_str == "":
                return None, None
                
            data = json.loads(json_str)
            political_score = data.get('Political', {}).get('score', None)
            stance_score = data.get('Stance', {}).get('score', None)
            
            return float(political_score) if political_score is not None else None, \
                   float(stance_score) if stance_score is not None else None
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None, None

    def load_topic_data(self, topic_key):
        """Load data for a specific topic"""
        topic_info = self.topics[topic_key]
        filename = "{}.csv".format(topic_info['filename'])
        filepath = os.path.join(self.annotation_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("File not found: {}".format(filepath))
        
        df = pd.read_csv(filepath)
        print("✓ {} data loaded: {} (shape: {})".format(topic_key, filename, df.shape))
        
        return df

    def calculate_pilot_statistics(self, df, topic_key):
        """Calculate pilot test statistics from first 300 samples"""
        pilot_df = df.iloc[:self.pilot_size].copy()
        
        political_row_averages = []
        stance_row_averages = []
        
        for idx, row in pilot_df.iterrows():
            row_political_scores = []
            row_stance_scores = []
            
            for combo in self.model_persona_combinations:
                if combo in pilot_df.columns:
                    pol_score, stance_score = self.parse_json_response(row[combo])
                    
                    if pol_score is not None:
                        row_political_scores.append(pol_score)
                    if stance_score is not None:
                        row_stance_scores.append(stance_score)
            
            # Calculate row-wise averages if we have scores from all personas
            if len(row_political_scores) == len(self.model_persona_combinations):
                political_row_averages.append(np.mean(row_political_scores))
            
            if len(row_stance_scores) == len(self.model_persona_combinations):
                stance_row_averages.append(np.mean(row_stance_scores))
        
        # Calculate statistics
        pol_std = np.std(political_row_averages, ddof=1) if len(political_row_averages) > 1 else 0
        stance_std = np.std(stance_row_averages, ddof=1) if len(stance_row_averages) > 1 else 0
        
        print("  {} pilot statistics: Political std={:.4f}, Stance std={:.4f}".format(topic_key, pol_std, stance_std))
        
        return {
            'political_std': pol_std,
            'stance_std': stance_std,
            'political_sample_size': len(political_row_averages),
            'stance_sample_size': len(stance_row_averages)
        }

    def calculate_required_sample_size(self, pilot_stats):
        """Calculate required sample size using formula: n = (Z * σ / E)²"""
        z_score = 1.960  # 95% confidence
        
        pol_required = math.ceil((z_score * pilot_stats['political_std'] / self.margin_error) ** 2)
        stance_required = math.ceil((z_score * pilot_stats['stance_std'] / self.margin_error) ** 2)
        
        # Use the maximum of the two
        max_required = max(pol_required, stance_required)
        
        return {
            'political_required': pol_required,
            'stance_required': stance_required,
            'max_required': max_required
        }

    def extract_scores_random_sampling(self, df, topic_key, required_sample_size):
        """Extract scores using random sampling from entire dataset (including pilot samples)"""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Get total available samples
        total_samples = len(df)
        
        # Determine actual sample size to use
        actual_sample_size = min(required_sample_size, total_samples)
        
        # Random sampling of row indices
        sampled_indices = np.random.choice(total_samples, size=actual_sample_size, replace=False)
        sampled_df = df.iloc[sampled_indices].copy()
        
        political_scores = []
        stance_scores = []
        
        for idx, row in sampled_df.iterrows():
            for combo in self.model_persona_combinations:
                if combo in sampled_df.columns:
                    pol_score, stance_score = self.parse_json_response(row[combo])
                    
                    if pol_score is not None:
                        political_scores.append(pol_score)
                    if stance_score is not None:
                        stance_scores.append(stance_score)
        
        print("  {} random sampling: Total {} rows, Political={}, Stance={} scores".format(
            topic_key, actual_sample_size, len(political_scores), len(stance_scores)))
        
        return {
            'political_scores': np.array(political_scores),
            'stance_scores': np.array(stance_scores),
            'sampled_indices': sampled_indices,
            'actual_sample_size': actual_sample_size
        }

    def perform_bias_test(self, scores, score_type, topic_key, threshold):
        """Perform TOST equivalence test for bias detection"""
        if len(scores) < 2:
            return None
        
        n = len(scores)
        sample_mean = np.mean(scores)
        sample_std = np.std(scores, ddof=1)
        sample_se = sample_std / np.sqrt(n)
        
        # Traditional t-test against 0 for statistical significance
        t_statistic, p_value_traditional = ttest_1samp(scores, 0)
        is_statistically_significant = p_value_traditional < self.alpha
        
        # TOST (Two One-Sided Tests) for Equivalence
        t1 = (sample_mean - (-threshold)) / sample_se
        p_value_t1 = 1 - stats.t.cdf(t1, n-1)
        
        t2 = (sample_mean - threshold) / sample_se
        p_value_t2 = stats.t.cdf(t2, n-1)
        
        tost_p_value = max(p_value_t1, p_value_t2)
        equivalence_shown = tost_p_value < self.alpha
        bias_detected = not equivalence_shown
        
        # 99% Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        margin_error = t_critical * sample_se
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        return {
            'topic': topic_key,
            'score_type': score_type,
            'threshold': threshold,
            'n': n,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic_traditional': t_statistic,
            'p_value_traditional': p_value_traditional,
            'is_statistically_significant': is_statistically_significant,
            'tost_p_value': tost_p_value,
            'bias_detected': bias_detected
        }

    def run_comprehensive_analysis(self):
        """Run comprehensive analysis for all topics"""
        print("\n=== Comprehensive Bias Analysis Started (Random Sampling) ===")
        
        all_results = []
        topic_sample_sizes = {}
        
        for topic_key in self.topics.keys():
            print("\n🔍 Analyzing {}...".format(topic_key))
            
            # Load data
            df = self.load_topic_data(topic_key)
            
            # Calculate pilot statistics for sample size estimation
            pilot_stats = self.calculate_pilot_statistics(df, topic_key)
            
            # Calculate required sample size
            sample_size_info = self.calculate_required_sample_size(pilot_stats)
            required_size = sample_size_info['max_required']
            topic_sample_sizes[topic_key] = required_size
            
            print("  Required sample size: {}".format(required_size))
            
            # Check if we have enough data
            if len(df) < required_size:
                print("  ⚠️ Insufficient data: Required={}, Available={}".format(required_size, len(df)))
                required_size = len(df)
                topic_sample_sizes[topic_key] = required_size
            
            # Extract scores using random sampling from entire dataset
            test_scores = self.extract_scores_random_sampling(df, topic_key, required_size)
            
            # Perform bias tests for both thresholds
            for threshold in self.neutrality_thresholds:
                # Political test
                pol_result = self.perform_bias_test(
                    test_scores['political_scores'], 'Political', topic_key, threshold
                )
                if pol_result:
                    all_results.append(pol_result)
                
                # Stance test
                stance_result = self.perform_bias_test(
                    test_scores['stance_scores'], 'Stance', topic_key, threshold
                )
                if stance_result:
                    all_results.append(stance_result)
        
        return pd.DataFrame(all_results), topic_sample_sizes

    def create_bias_visualization(self, results_df, topic_sample_sizes):
        """Create publication-quality bias visualization with topic separation lines"""
        # Set publication style with larger fonts
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.edgecolor': 'black'
        })
        
        # Set up the figure with proper aspect ratio for publication
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Publication-quality colors
        colors = {
            'Political': '#E74C3C',  # Professional red
            'Stance': '#3498DB'      # Professional blue
        }
        
        # Topic ordering (reverse for proper display)
        topics_ordered = list(reversed(list(self.topics.keys())))
        n_topics = len(topics_ordered)
        
        # Y-axis positioning - more space between topics to prevent overlap
        y_spacing = 1.2
        y_positions = {}
        for i, topic_key in enumerate(topics_ordered):
            y_positions[topic_key] = i * y_spacing
        
        # Plot neutrality zones with professional styling
        for threshold in self.neutrality_thresholds:
            if threshold == 0.10:
                ax.axvspan(-threshold, threshold, alpha=0.12, color='#3498DB', 
                          label='±0.10 Neutrality Zone', zorder=1)
            else:
                ax.axvspan(-threshold, threshold, alpha=0.08, color='#95A5A6', 
                          label='±0.20 Neutrality Zone', zorder=1)
        
        # Add central reference line
        ax.axvline(x=0, color='#2C3E50', linestyle='-', alpha=0.6, linewidth=1.2, zorder=2)
        
        # Add horizontal separation lines between topics
        for i in range(n_topics - 1):
            y_sep = (i + 0.5) * y_spacing
            ax.axhline(y=y_sep, color='#BDC3C7', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
        
        # Get results for visualization (using 0.10 threshold)
        threshold_10_results = results_df[results_df['threshold'] == 0.10]
        
        # Plot data for each topic
        for topic_key in topics_ordered:
            y_base = y_positions[topic_key]
            topic_results = threshold_10_results[threshold_10_results['topic'] == topic_key]
            
            # Group results by score type
            political_data = topic_results[topic_results['score_type'] == 'Political']
            stance_data = topic_results[topic_results['score_type'] == 'Stance']
            
            # Draw horizontal line for the topic (baseline)
            ax.axhline(y=y_base, color='#34495E', linestyle='-', alpha=0.3, linewidth=1.5, zorder=2)
            
            # Plot Political Score
            if len(political_data) > 0:
                row = political_data.iloc[0]
                mean_val = row['sample_mean']
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                
                # Draw confidence interval as horizontal line
                ax.plot([ci_lower, ci_upper], [y_base, y_base], 
                       color=colors['Political'], linewidth=3, alpha=0.4, zorder=3)
                
                # Mark mean value with larger marker
                ax.plot(mean_val, y_base, marker='o', color=colors['Political'], 
                       markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=5)
                
                # Add mean value text above the point with more spacing
                ax.text(mean_val, y_base + 0.25, '{:.3f}'.format(mean_val), 
                        va='bottom', ha='center', fontsize=13, 
                        color=colors['Political'], fontweight='bold')
            
            # Plot Stance Score
            if len(stance_data) > 0:
                row = stance_data.iloc[0]
                mean_val = row['sample_mean']
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                
                # Draw confidence interval as horizontal line (slightly offset)
                ax.plot([ci_lower, ci_upper], [y_base - 0.05, y_base - 0.05], 
                       color=colors['Stance'], linewidth=3, alpha=0.4, zorder=3)
                
                # Mark mean value with square marker
                ax.plot(mean_val, y_base - 0.05, marker='s', color=colors['Stance'], 
                       markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=5)
                
                # Add mean value text below the point with more spacing
                ax.text(mean_val, y_base - 0.3, '{:.3f}'.format(mean_val), 
                       va='top', ha='center', fontsize=13, 
                       color=colors['Stance'], fontweight='bold')
        
        # Customize axes with more space for top labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.6, (n_topics - 1) * y_spacing + 1.0)
        
        # Set y-axis labels with topic names and sample sizes
        y_ticks = [y_positions[topic] for topic in topics_ordered]
        y_labels = []
        for topic_key in topics_ordered:
            sample_size = topic_sample_sizes.get(topic_key, 'N/A')
            display_name = self.topics[topic_key]['display_name']
            y_labels.append("{}\n(n={})".format(display_name, sample_size))
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=14)
        
        # Set x-axis with appropriate ticks
        ax.set_xlabel('Average Score', fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(-1.0, 1.1, 0.2))
        ax.set_xticklabels(['{:.1f}'.format(x) for x in np.arange(-1.0, 1.1, 0.2)])
        
        # Add directional labels at the top with more spacing to avoid overlap
        top_label_y = (n_topics - 1) * y_spacing + 0.7  # Increased spacing
        ax.text(-0.4, top_label_y, 
               'Left/Against', ha='center', va='center', 
               fontsize=15, style='italic', color='black')
        ax.text(0, top_label_y, 
               'Neutral', ha='center', va='center', 
               fontsize=15, style='italic', color='black', fontweight='bold')
        ax.text(0.4, top_label_y, 
               'Right/Support', ha='center', va='center', 
               fontsize=15, style='italic', color='black')
        
        # Add title
        # ax.set_title('Political and Stance Bias Analysis Across Topics\n(99% Confidence Intervals)', 
        #             fontsize=15, fontweight='bold', pad=40)
        
        # Create professional legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color=colors['Political'], linewidth=0, 
                      markersize=12, markeredgecolor='white', markeredgewidth=2,
                      label='Political Score'),
            plt.Line2D([0], [0], marker='s', color=colors['Stance'], linewidth=0, 
                      markersize=12, markeredgecolor='white', markeredgewidth=2,
                      label='Stance Score'),
            plt.Line2D([0], [0], color=colors['Political'], linewidth=3, alpha=0.4,
                      label='99% Confidence Interval'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#3498DB', alpha=0.12, 
                         label='±0.10 Neutrality Zone'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#95A5A6', alpha=0.08, 
                         label='±0.20 Neutrality Zone'),
            plt.Line2D([0], [0], color='#34495E', linestyle='-', alpha=0.3, linewidth=1.5,
                      label='Topic Baseline'),
            plt.Line2D([0], [0], color='#BDC3C7', linestyle='--', alpha=0.5, linewidth=0.8,
                      label='Topic Separator')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.2, 1), fontsize=12)
        
        # Final layout adjustments
        plt.tight_layout()
        
        # Save with publication quality
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.fig_dir, 'publication_bias_analysis_{}.png'.format(timestamp))
        plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        plt.savefig(os.path.join(self.fig_dir, 'publication_bias_analysis_latest.png'), 
                   dpi=600, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        
        # Also save as high-quality PDF for publications
        # pdf_path = os.path.join(self.fig_dir, 'publication_bias_analysis_{}.pdf'.format(timestamp))
        # plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
        #            edgecolor='none', format='pdf')
        # plt.savefig(os.path.join(self.fig_dir, 'publication_bias_analysis_latest.pdf'), 
        #            bbox_inches='tight', facecolor='white', 
        #            edgecolor='none', format='pdf')
        
        print("\n📊 Publication-quality visualization saved:")
        print("   PNG: {}".format(fig_path))
        # print("   PDF: {}".format(pdf_path))
        
        # Reset matplotlib rcParams to default
        plt.rcParams.update(plt.rcParamsDefault)
        
        return fig_path

    def create_topic_summary_table(self, results_df, topic_sample_sizes):
        """Create detailed summary table for each topic with bias detection results"""
        summary_data = []
        
        for topic_key in self.topics.keys():
            topic_results = results_df[results_df['topic'] == topic_key]
            
            if len(topic_results) == 0:
                continue
            
            # Initialize row data
            row_data = {
                'Topic': self.topics[topic_key]['display_name'],
                'Topic_Key': topic_key,
                'Sample_Size': topic_sample_sizes.get(topic_key, 'N/A'),
                'Political_Mean': None,
                'Stance_Mean': None,
                'Political_10_PValue': None,
                'Political_10_TOST_PValue': None,
                'Political_10_Bias_Status': None,
                'Political_10_CI_Lower': None,
                'Political_10_CI_Upper': None,
                'Stance_10_PValue': None,
                'Stance_10_TOST_PValue': None,
                'Stance_10_Bias_Status': None,
                'Stance_10_CI_Lower': None,
                'Stance_10_CI_Upper': None,
                'Political_20_PValue': None,
                'Political_20_TOST_PValue': None,
                'Political_20_Bias_Status': None,
                'Political_20_CI_Lower': None,
                'Political_20_CI_Upper': None,
                'Stance_20_PValue': None,
                'Stance_20_TOST_PValue': None,
                'Stance_20_Bias_Status': None,
                'Stance_20_CI_Lower': None,
                'Stance_20_CI_Upper': None
            }
            
            # Extract data for each threshold and score type
            for threshold in self.neutrality_thresholds:
                threshold_results = topic_results[topic_results['threshold'] == threshold]
                threshold_str = str(int(threshold * 100))  # 0.10 -> "10", 0.20 -> "20"
                
                for _, row in threshold_results.iterrows():
                    score_type = row['score_type']
                    mean_val = row['sample_mean']
                    p_val_traditional = row['p_value_traditional']
                    p_val_tost = row['tost_p_value']
                    bias_detected = row['bias_detected']
                    ci_lower = row['ci_lower']
                    ci_upper = row['ci_upper']
                    
                    # Determine bias status
                    if bias_detected:
                        if mean_val > 0:
                            if score_type == 'Political':
                                bias_status = "Right-biased"
                            else:  # Stance
                                bias_status = "Support-biased"
                        else:
                            if score_type == 'Political':
                                bias_status = "Left-biased"
                            else:  # Stance
                                bias_status = "Against-biased"
                    else:
                        bias_status = "Neutral"
                    
                    # Store mean values (same across thresholds)
                    if score_type == 'Political':
                        row_data['Political_Mean'] = mean_val
                    else:  # Stance
                        row_data['Stance_Mean'] = mean_val
                    
                    # Store threshold-specific data
                    if score_type == 'Political':
                        row_data['Political_{}_PValue'.format(threshold_str)] = p_val_traditional
                        row_data['Political_{}_TOST_PValue'.format(threshold_str)] = p_val_tost
                        row_data['Political_{}_Bias_Status'.format(threshold_str)] = bias_status
                        row_data['Political_{}_CI_Lower'.format(threshold_str)] = ci_lower
                        row_data['Political_{}_CI_Upper'.format(threshold_str)] = ci_upper
                    else:  # Stance
                        row_data['Stance_{}_PValue'.format(threshold_str)] = p_val_traditional
                        row_data['Stance_{}_TOST_PValue'.format(threshold_str)] = p_val_tost
                        row_data['Stance_{}_Bias_Status'.format(threshold_str)] = bias_status
                        row_data['Stance_{}_CI_Lower'.format(threshold_str)] = ci_lower
                        row_data['Stance_{}_CI_Upper'.format(threshold_str)] = ci_upper
            
            summary_data.append(row_data)
        
        return pd.DataFrame(summary_data)

    def save_results(self, results_df, topic_sample_sizes):
        """Save analysis results including topic summary table"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.analysis_results_dir, 
                                   'integrated_bias_analysis_{}.csv'.format(timestamp))
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # Save sample size summary
        sample_size_df = pd.DataFrame([
            {'Topic': topic, 'Required_Sample_Size': size} 
            for topic, size in topic_sample_sizes.items()
        ])
        
        sample_size_path = os.path.join(self.analysis_results_dir, 
                                       'sample_sizes_{}.csv'.format(timestamp))
        sample_size_df.to_csv(sample_size_path, index=False, encoding='utf-8-sig')
        
        # Create and save topic summary table
        topic_summary_df = self.create_topic_summary_table(results_df, topic_sample_sizes)
        topic_summary_path = os.path.join(self.analysis_results_dir, 
                                         'topic_summary_{}.csv'.format(timestamp))
        topic_summary_df.to_csv(topic_summary_path, index=False, encoding='utf-8-sig')
        
        print("📄 Results saved:")
        print("  Detailed results: {}".format(results_path))
        print("  Sample sizes: {}".format(sample_size_path))
        print("  Topic summary: {}".format(topic_summary_path))
        
        return results_path, sample_size_path, topic_summary_path

    def print_summary_report(self, results_df, topic_sample_sizes):
        """Print comprehensive summary report"""
        print("\n{}".format("="*80))
        print("Integrated Bias Analysis Summary Report")
        print("{}".format("="*80))
        
        print("\n1. Sample Size Summary:")
        print("{}".format("-"*50))
        for topic, size in topic_sample_sizes.items():
            print("  {}: {} samples".format(topic, size))
        
        print("\n2. Bias Testing Results (δ = ±0.20):")
        print("{}".format("-"*50))
        
        threshold_20_results = results_df[results_df['threshold'] == 0.20]
        
        for topic_key in self.topics.keys():
            topic_results = threshold_20_results[threshold_20_results['topic'] == topic_key]
            print("\n📊 {}:".format(topic_key))
            
            for _, row in topic_results.iterrows():
                score_type = row['score_type']
                mean = row['sample_mean']
                bias_detected = row['bias_detected']
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                
                status = "Bias detected" if bias_detected else "Neutral"
                print("  {}: Mean={:.4f}, CI=[{:.4f}, {:.4f}], {}".format(score_type, mean, ci_lower, ci_upper, status))
        
        print("\n3. Overall Bias Detection Rate:")
        print("{}".format("-"*50))
        
        total_tests = len(threshold_20_results)
        biased_tests = threshold_20_results['bias_detected'].sum()
        bias_rate = biased_tests / total_tests if total_tests > 0 else 0
        
        print("  Total tests: {}".format(total_tests))
        print("  Bias detected: {}".format(biased_tests))
        print("  Bias detection rate: {:.1%}".format(bias_rate))

    def run_analysis(self):
        """Run the complete integrated analysis"""
        print("Starting integrated bias analysis...")
        
        # Run comprehensive analysis
        results_df, topic_sample_sizes = self.run_comprehensive_analysis()
        
        # Create visualization
        fig_path = self.create_bias_visualization(results_df, topic_sample_sizes)
        
        # Save results
        results_path, sample_size_path, topic_summary_path = self.save_results(results_df, topic_sample_sizes)
        
        # Print summary report
        self.print_summary_report(results_df, topic_sample_sizes)
        
        print("\n✅ Integrated bias analysis completed!")
        return results_df, topic_sample_sizes, fig_path, topic_summary_path


if __name__ == '__main__':
    # Initialize and run analysis
    analyzer = IntegratedBiasAnalysis()
    
    try:
        results, sample_sizes, figure_path, topic_summary_path = analyzer.run_analysis()
        print("\n🎯 Analysis completed! Figure file: {}".format(figure_path))
        print("📊 Topic summary file: {}".format(topic_summary_path))
        
    except Exception as e:
        print("\n❌ Error during analysis: {}".format(e))
        import traceback
        traceback.print_exc() 