import os
import pandas as pd
import json
import numpy as np
import math
from scipy import stats
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

base_filename = "annotated_death_penalty_20250805_024453"
sample_size = 1332
base_size = 300
class BiasDetectionTest:
    def __init__(self, base_filename=base_filename):
        self.base_filename = base_filename
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.annotation_dir = os.path.join(self.current_dir, '../annotation_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        
        # Create analysis results directory if it doesn't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        
        # Claude model only (high consistency: 98-99%)
        self.models = ['claude-sonnet-4-20250514','gpt-4.1']
        self.personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
        self.model_persona_combinations = []
        
        for model in self.models:
            for persona in self.personas:
                self.model_persona_combinations.append("{}_{}" .format(model, persona))
        
        # Pilot test and sample size calculation parameters
        self.pilot_size = 300  # First 300 samples for pilot test
        self.confidence_level = 0.95
        self.margin_error = 0.03
        
        # Statistical testing parameters
        self.alpha = 0.01  # 99% confidence level
        self.neutrality_thresholds = [0.05, 0.10, 0.15, 0.20]  # Different neutrality ranges
        
        print("Bias Detection Test: TOST Equivalence Testing with Pilot-based Sample Size Calculation")
        print("Sample file: {}.csv" .format(base_filename))
        print("Model: Claude Sonnet 4 (High Consistency: 98-99%)")
        print("Pilot test size: {}".format(self.pilot_size))
        print("Sample size calculation: {}% confidence, ±{}% margin of error".format(
            int(self.confidence_level*100), int(self.margin_error*100)))
        print("\nTOST Hypothesis Framework:")
        print("H₀₁: μ ≤ -δ (Lower bound test)")
        print("H₀₂: μ ≥ +δ (Upper bound test)")
        print("H₁: -δ < μ < +δ (Equivalent/Neutral)")
        print("Bias Detection: If TOST fails → Bias detected")
        print("Neutrality thresholds (δ): {}".format(self.neutrality_thresholds))
        print("Significance level: α = {} (99% confidence)".format(self.alpha))

    def load_annotation_data(self):
        """Load annotation data from the first repetition file"""
        filename = "{}.csv".format(self.base_filename)
        filepath = os.path.join(self.annotation_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("Annotation file not found: {}".format(filepath))
        
        df = pd.read_csv(filepath)
        # df = df[base_size:base_size+sample_size]
        print("\n✓ Loaded annotation data: {}".format(filename))
        print("Dataset shape: {}".format(df.shape))
        print("Sample size: {} articles".format(len(df)))
        
        return df

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

    def calculate_pilot_statistics(self, df):
        """Calculate pilot test statistics from first 300 samples"""
        pilot_df = df.iloc[:self.pilot_size].copy()
        
        political_row_averages = []
        stance_row_averages = []
        
        print("\nCalculating pilot test statistics from first {} samples...".format(self.pilot_size))
        
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
        
        print("  Pilot statistics: Political std={:.4f}, Stance std={:.4f}".format(pol_std, stance_std))
        print("  Valid pilot samples: Political={}, Stance={}".format(
            len(political_row_averages), len(stance_row_averages)))
        
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
        
        print("\nSample size calculation results:")
        print("  Political required: {} samples".format(pol_required))
        print("  Stance required: {} samples".format(stance_required))
        print("  Maximum required: {} samples".format(max_required))
        
        return {
            'political_required': pol_required,
            'stance_required': stance_required,
            'max_required': max_required
        }

    def extract_test_scores(self, df, required_sample_size):
        """Extract scores for bias testing, excluding pilot samples"""
        start_idx = self.pilot_size
        end_idx = start_idx + required_sample_size
        
        # Check if we have enough data
        available_after_pilot = len(df) - self.pilot_size
        if available_after_pilot < required_sample_size:
            print("  ⚠️ Data shortage: required={}, available={}".format(
                required_sample_size, available_after_pilot))
            end_idx = len(df)  # Use all available data
            actual_sample_size = available_after_pilot
        else:
            actual_sample_size = required_sample_size
        
        test_df = df.iloc[start_idx:end_idx].copy()
        
        print("\nExtracting test scores from samples {} to {}...".format(start_idx, end_idx-1))
        print("  Actual test sample size: {} articles".format(actual_sample_size))
        
        political_scores = []
        stance_scores = []
        
        for idx, row in test_df.iterrows():
            for combo in self.model_persona_combinations:
                if combo in test_df.columns:
                    pol_score, stance_score = self.parse_json_response(row[combo])
                    
                    if pol_score is not None:
                        political_scores.append(pol_score)
                    if stance_score is not None:
                        stance_scores.append(stance_score)
        
        print("  Test data: Political={}, Stance={} scores".format(
            len(political_scores), len(stance_scores)))
        
        return {
            'political_scores': np.array(political_scores),
            'stance_scores': np.array(stance_scores),
            'actual_sample_size': actual_sample_size
        }

    def extract_scores_by_combination(self, df):
        """Extract scores for each Claude model-persona combination"""
        print("\nExtracting scores for bias detection testing...")
        
        scores_data = {}
        
        for combo in self.model_persona_combinations:
            if combo in df.columns:
                political_scores = []
                stance_scores = []
                
                for idx, row in df.iterrows():
                    pol_score, stance_score = self.parse_json_response(row[combo])
                    
                    if pol_score is not None:
                        political_scores.append(pol_score)
                    if stance_score is not None:
                        stance_scores.append(stance_score)
                
                scores_data[combo] = {
                    'political_scores': np.array(political_scores),
                    'stance_scores': np.array(stance_scores)
                }
                
                print("  {}: Political={}, Stance={} valid scores".format(
                    combo, len(political_scores), len(stance_scores)))
        
        return scores_data

    def calculate_effect_size(self, scores, population_mean=0.0):
        """Calculate Cohen's d effect size"""
        if len(scores) == 0:
            return None
        
        sample_mean = np.mean(scores)
        sample_std = np.std(scores, ddof=1)
        
        if sample_std == 0:
            return float('inf') if sample_mean != population_mean else 0
        
        cohens_d = (sample_mean - population_mean) / sample_std
        return cohens_d

    def interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size"""
        if effect_size is None:
            return "Cannot calculate"
        
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "Negligible"
        elif abs_effect < 0.5:
            return "Small"
        elif abs_effect < 0.8:
            return "Medium"
        else:
            return "Large"

    def perform_bias_detection_test(self, scores, score_type, combination_name, threshold):
        """
        Perform TOST (Two One-Sided Tests) for Equivalence Testing
        
        TOST Equivalence Test:
        H₀₁: μ ≤ -δ (Lower bound test)
        H₀₂: μ ≥ +δ (Upper bound test)  
        H₁: -δ < μ < +δ (Equivalent/Neutral)
        
        For bias detection, we reverse the interpretation:
        - If TOST shows equivalence → Neutral (no bias)
        - If TOST rejects equivalence → Biased
        """
        if len(scores) < 2:
            return None
        
        # Basic statistics
        n = len(scores)
        sample_mean = np.mean(scores)
        sample_std = np.std(scores, ddof=1)
        sample_se = sample_std / np.sqrt(n)
        
        # Check if sample mean falls within neutrality range
        is_within_neutral_range = abs(sample_mean) < threshold
        
        # Traditional t-test against 0 for statistical significance
        t_statistic, p_value_traditional = ttest_1samp(scores, 0)
        is_statistically_significant = p_value_traditional < self.alpha
        
        # TOST (Two One-Sided Tests) for Equivalence
        # Test 1: H₀₁: μ ≤ -δ vs H₁₁: μ > -δ
        t1 = (sample_mean - (-threshold)) / sample_se
        p_value_t1 = 1 - stats.t.cdf(t1, n-1)
        
        # Test 2: H₀₂: μ ≥ +δ vs H₁₂: μ < +δ  
        t2 = (sample_mean - threshold) / sample_se
        p_value_t2 = stats.t.cdf(t2, n-1)
        
        # TOST result: equivalence if both tests reject their null hypotheses
        tost_p_value = max(p_value_t1, p_value_t2)  # More conservative approach
        equivalence_shown = tost_p_value < self.alpha
        
        # For bias detection: reverse the TOST interpretation
        # If TOST shows equivalence → Neutral (no bias detected)
        # If TOST fails to show equivalence → Bias detected
        bias_detected = not equivalence_shown
        
        # Degrees of freedom
        df = n - 1
        
        # 99% Confidence interval
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin_error = t_critical * sample_se
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        # Effect size
        effect_size = self.calculate_effect_size(scores, 0)
        effect_interpretation = self.interpret_effect_size(effect_size)
        
        # Overall conclusion based on TOST
        if bias_detected:  # TOST failed to show equivalence
            if sample_mean > 0:
                bias_direction = "Positive"
                bias_type = "Right/Support" if score_type == "Political" else "Support"
            else:
                bias_direction = "Negative" 
                bias_type = "Left/Against" if score_type == "Political" else "Against"
            conclusion = "Bias Detected - TOST failed to show equivalence ({})".format(bias_direction)
        else:  # TOST showed equivalence
            conclusion = "Neutral - TOST equivalence shown (within ±{} range)".format(threshold)
            bias_type = "None"
        
        return {
            'combination': combination_name,
            'score_type': score_type,
            'threshold': threshold,
            'n': n,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'sample_se': sample_se,
            
            # Traditional test results
            't_statistic_traditional': t_statistic,
            'p_value_traditional': p_value_traditional,
            'is_statistically_significant': is_statistically_significant,
            
            # TOST results
            't1_statistic': t1,
            't2_statistic': t2,
            'p_value_t1': p_value_t1,
            'p_value_t2': p_value_t2,
            'tost_p_value': tost_p_value,
            'equivalence_shown': equivalence_shown,
            'p_value_threshold': tost_p_value,  # For compatibility with existing code
            'bias_detected': bias_detected,
            
            # Range assessment
            'is_within_neutral_range': is_within_neutral_range,
            'abs_mean': abs(sample_mean),
            
            # Confidence intervals and effect size
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_size_cohens_d': effect_size,
            'effect_interpretation': effect_interpretation,
            
            # Conclusions
            'conclusion': conclusion,
            'bias_type': bias_type,
            'alpha': self.alpha
        }

    def perform_comprehensive_analysis(self, scores_data):
        """Perform comprehensive bias detection analysis for all thresholds"""
        print("\nPerforming bias detection tests...")
        print("{}".format("="*80))
        
        all_results = []
        
        # Test each threshold
        for threshold in self.neutrality_thresholds:
            print("\n🔍 TESTING NEUTRALITY THRESHOLD: δ = ±{}".format(threshold))
            print("{}".format("─"*60))
            
            # Test each Claude model-persona combination
            for combo, data in scores_data.items():
                print("\nAnalyzing: {} (δ = {})".format(combo, threshold))
                print("{}".format("-"*40))
                
                # Political scores analysis
                pol_scores = data['political_scores']
                if len(pol_scores) > 0:
                    pol_result = self.perform_bias_detection_test(pol_scores, 'Political', combo, threshold)
                    if pol_result:
                        all_results.append(pol_result)
                        
                        p_val_trad = pol_result['p_value_traditional']
                        p_val_tost = pol_result['tost_p_value']
                        p_str_trad = "p={:.2e}".format(p_val_trad) if p_val_trad < 0.001 else "p={:.6f}".format(p_val_trad)
                        p_str_tost = "p={:.2e}".format(p_val_tost) if p_val_tost < 0.001 else "p={:.6f}".format(p_val_tost)
                        
                        print("Political: mean={:.4f}, traditional {}, TOST {}".format(
                            pol_result['sample_mean'], p_str_trad, p_str_tost))
                        print("  → {}".format(pol_result['conclusion']))
                
                # Stance scores analysis
                stance_scores = data['stance_scores']
                if len(stance_scores) > 0:
                    stance_result = self.perform_bias_detection_test(stance_scores, 'Stance', combo, threshold)
                    if stance_result:
                        all_results.append(stance_result)
                        
                        p_val_trad = stance_result['p_value_traditional']
                        p_val_tost = stance_result['tost_p_value']
                        p_str_trad = "p={:.2e}".format(p_val_trad) if p_val_trad < 0.001 else "p={:.6f}".format(p_val_trad)
                        p_str_tost = "p={:.2e}".format(p_val_tost) if p_val_tost < 0.001 else "p={:.6f}".format(p_val_tost)
                        
                        print("Stance:    mean={:.4f}, traditional {}, TOST {}".format(
                            stance_result['sample_mean'], p_str_trad, p_str_tost))
                        print("  → {}".format(stance_result['conclusion']))
        
        return pd.DataFrame(all_results)

    def calculate_aggregated_analysis(self, test_scores_data):
        """Calculate aggregated analysis using pilot-test based sample"""
        print("\nCalculating aggregated analysis from test data...")
        
        # Use test scores directly (already aggregated across all personas)
        all_political_scores = test_scores_data['political_scores']
        all_stance_scores = test_scores_data['stance_scores']
        
        print("  Test data: Political={}, Stance={} scores".format(
            len(all_political_scores), len(all_stance_scores)))
        
        aggregated_results = []
        
        # Test each threshold for aggregated data
        for threshold in self.neutrality_thresholds:
            # Aggregated Political analysis
            if len(all_political_scores) > 0:
                pol_result = self.perform_bias_detection_test(
                    all_political_scores, 'Political_Aggregated', 'Claude_All_Personas', threshold
                )
                if pol_result:
                    aggregated_results.append(pol_result)
            
            # Aggregated Stance analysis
            if len(all_stance_scores) > 0:
                stance_result = self.perform_bias_detection_test(
                    all_stance_scores, 'Stance_Aggregated', 'Claude_All_Personas', threshold
                )
                if stance_result:
                    aggregated_results.append(stance_result)
        
        return pd.DataFrame(aggregated_results)
    
    def apply_multiple_testing_corrections(self, results_df):
        """Apply multiple testing corrections to p-values by threshold
        
        Note: For TOST, we need to be careful because:
        - Small p-value = good equivalence = neutral (opposite of usual interpretation)
        - Large p-value = poor equivalence = biased
        
        We apply corrections to the original TOST p-values but interpret them correctly.
        """
        if len(results_df) == 0:
            return results_df
        
        # Initialize corrected columns
        results_df['p_value_bonferroni'] = results_df['p_value_threshold'].copy()
        results_df['p_value_fdr'] = results_df['p_value_threshold'].copy()
        results_df['bias_detected_bonferroni'] = results_df['bias_detected'].copy()
        results_df['bias_detected_fdr'] = results_df['bias_detected'].copy()
        results_df['bonferroni_alpha'] = self.alpha
        results_df['fdr_alpha'] = self.alpha
        
        # Apply corrections separately for each threshold
        for threshold in self.neutrality_thresholds:
            threshold_mask = results_df['threshold'] == threshold
            threshold_data = results_df[threshold_mask]
            
            if len(threshold_data) == 0:
                continue
            
            # Extract p-values for this threshold
            p_values = threshold_data['p_value_threshold'].values
            n_tests = len(p_values)
            
            # Apply Bonferroni correction
            p_bonferroni = p_values * n_tests
            p_bonferroni = np.minimum(p_bonferroni, 1.0)  # Cap at 1.0
            
            # Apply FDR correction (Benjamini-Hochberg)
            reject_fdr, p_fdr, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.alpha, method='fdr_bh'
            )
            
            # Update results for this threshold
            results_df.loc[threshold_mask, 'p_value_bonferroni'] = p_bonferroni
            results_df.loc[threshold_mask, 'p_value_fdr'] = p_fdr
            results_df.loc[threshold_mask, 'bonferroni_alpha'] = self.alpha / n_tests
            results_df.loc[threshold_mask, 'fdr_alpha'] = self.alpha
            
            # CRITICAL: For TOST, bias detection logic is REVERSED
            # Small corrected p-value = good equivalence = neutral (no bias)
            # Large corrected p-value = poor equivalence = biased
            results_df.loc[threshold_mask, 'bias_detected_bonferroni'] = p_bonferroni >= self.alpha
            results_df.loc[threshold_mask, 'bias_detected_fdr'] = ~reject_fdr  # Reverse FDR decision
            
            print(f"\nMultiple Testing Correction for δ = ±{threshold}:")
            print(f"Number of tests: {n_tests}")
            print(f"Original α: {self.alpha}")
            print(f"Bonferroni corrected α: {self.alpha / n_tests:.6f}")
            print(f"FDR corrected α: {self.alpha}")
        
        return results_df

    def create_summary_table(self, aggregated_results, correction_type='none'):
        """Create detailed summary table with comprehensive statistics"""
        summary_data = []
        
        # Determine which columns to use based on correction type
        if correction_type == 'bonferroni':
            bias_col = 'bias_detected_bonferroni'
            p_val_col = 'p_value_bonferroni'
        elif correction_type == 'fdr':
            bias_col = 'bias_detected_fdr'
            p_val_col = 'p_value_fdr'
        else:  # no correction
            bias_col = 'bias_detected'
            p_val_col = 'p_value_threshold'
        
        for threshold in self.neutrality_thresholds:
            threshold_results = aggregated_results[aggregated_results['threshold'] == threshold]
            
            # Initialize row data with comprehensive statistics
            row_data = {
                'Neutrality_Range': "±{}%".format(int(threshold * 100)),
                'Sample_Size': '',
                'Political_Mean': '',
                'Political_StdDev': '',
                'Political_Effect_Size': '',
                'Political_CI_Lower': '',
                'Political_CI_Upper': '',
                'Political_Traditional_P': '',
                'Political_TOST_P': '',
                'Political_Status': '',
                'Stance_Mean': '',
                'Stance_StdDev': '',
                'Stance_Effect_Size': '',
                'Stance_CI_Lower': '',
                'Stance_CI_Upper': '',
                'Stance_Traditional_P': '',
                'Stance_TOST_P': '',
                'Stance_Status': '',
                'Total_Bias_Detection_Rate': '',
                'Conclusion': ''
            }
            
            bias_detected_count = 0
            total_tests = 0
            
            for _, row in threshold_results.iterrows():
                score_type = row['score_type']
                
                # Use appropriate columns based on correction type
                if bias_col in row:
                    bias_detected = row[bias_col]
                    p_val_thresh = row[p_val_col]
                else:
                    bias_detected = row['bias_detected']
                    p_val_thresh = row['p_value_threshold']
                
                # Format p-values
                p_trad = row['p_value_traditional']
                p_trad_str = "{:.2e}".format(p_trad) if p_trad < 0.001 else "{:.6f}".format(p_trad)
                p_tost_str = "{:.2e}".format(p_val_thresh) if p_val_thresh < 0.001 else "{:.6f}".format(p_val_thresh)
                
                # Determine status
                if bias_detected:
                    status = "Bias"
                    bias_detected_count += 1
                else:
                    status = "Neutral"
                
                # Fill in the appropriate columns
                if 'Political' in score_type:
                    row_data['Sample_Size'] = str(int(row['n']))
                    row_data['Political_Mean'] = "{:.4f}".format(row['sample_mean'])
                    row_data['Political_StdDev'] = "{:.4f}".format(row['sample_std'])
                    row_data['Political_Effect_Size'] = "{:.3f} ({})".format(
                        row['effect_size_cohens_d'], row['effect_interpretation'])
                    row_data['Political_CI_Lower'] = "{:.4f}".format(row['ci_lower'])
                    row_data['Political_CI_Upper'] = "{:.4f}".format(row['ci_upper'])
                    row_data['Political_Traditional_P'] = p_trad_str
                    row_data['Political_TOST_P'] = p_tost_str
                    row_data['Political_Status'] = status
                    
                elif 'Stance' in score_type:
                    row_data['Stance_Mean'] = "{:.4f}".format(row['sample_mean'])
                    row_data['Stance_StdDev'] = "{:.4f}".format(row['sample_std'])
                    row_data['Stance_Effect_Size'] = "{:.3f} ({})".format(
                        row['effect_size_cohens_d'], row['effect_interpretation'])
                    row_data['Stance_CI_Lower'] = "{:.4f}".format(row['ci_lower'])
                    row_data['Stance_CI_Upper'] = "{:.4f}".format(row['ci_upper'])
                    row_data['Stance_Traditional_P'] = p_trad_str
                    row_data['Stance_TOST_P'] = p_tost_str
                    row_data['Stance_Status'] = status
                
                total_tests += 1
            
            # Calculate overall rate
            if total_tests > 0:
                bias_rate = bias_detected_count / total_tests
                row_data['Total_Bias_Detection_Rate'] = "{:.1%}".format(bias_rate)
                
                # Determine conclusion
                if bias_rate >= 0.5:
                    row_data['Conclusion'] = "Bias exists"
                else:
                    row_data['Conclusion'] = "Neutral"
            
            summary_data.append(row_data)
        
        return pd.DataFrame(summary_data)

    def create_persona_summary_tables(self, individual_results, correction_type='none'):
        """Create summary tables for each persona"""
        persona_tables = {}
        
        # Determine which columns to use based on correction type
        if correction_type == 'bonferroni':
            bias_col = 'bias_detected_bonferroni'
            p_val_col = 'p_value_bonferroni'
        elif correction_type == 'fdr':
            bias_col = 'bias_detected_fdr'
            p_val_col = 'p_value_fdr'
        else:  # no correction
            bias_col = 'bias_detected'
            p_val_col = 'p_value_threshold'
        
        # Extract unique personas from combinations
        personas = []
        for combo in self.model_persona_combinations:
            persona = combo.split('_')[-1]
            if persona not in personas:
                personas.append(persona)
        
        for persona in personas:
            persona_data = []
            
            for threshold in self.neutrality_thresholds:
                # Filter results for this persona and threshold
                persona_results = individual_results[
                    (individual_results['combination'].str.contains(persona)) & 
                    (individual_results['threshold'] == threshold)
                ]
                
                # Initialize row data with comprehensive statistics
                row_data = {
                    'Neutrality_Range': "±{}%".format(int(threshold * 100)),
                    'Sample_Size': '',
                    'Political_Mean': '',
                    'Political_StdDev': '',
                    'Political_Effect_Size': '',
                    'Political_CI_Lower': '',
                    'Political_CI_Upper': '',
                    'Political_Traditional_P': '',
                    'Political_TOST_P': '',
                    'Political_Status': '',
                    'Stance_Mean': '',
                    'Stance_StdDev': '',
                    'Stance_Effect_Size': '',
                    'Stance_CI_Lower': '',
                    'Stance_CI_Upper': '',
                    'Stance_Traditional_P': '',
                    'Stance_TOST_P': '',
                    'Stance_Status': '',
                    'Total_Bias_Detection_Rate': '',
                    'Conclusion': ''
                }
                
                bias_detected_count = 0
                total_tests = 0
                
                for _, row in persona_results.iterrows():
                    score_type = row['score_type']
                    
                    # Use appropriate columns based on correction type
                    if bias_col in row:
                        bias_detected = row[bias_col]
                        p_val_thresh = row[p_val_col]
                    else:
                        bias_detected = row['bias_detected']
                        p_val_thresh = row['p_value_threshold']
                    
                    # Format p-values
                    p_trad = row['p_value_traditional']
                    p_trad_str = "{:.2e}".format(p_trad) if p_trad < 0.001 else "{:.6f}".format(p_trad)
                    p_tost_str = "{:.2e}".format(p_val_thresh) if p_val_thresh < 0.001 else "{:.6f}".format(p_val_thresh)
                    
                    # Determine status
                    if bias_detected:
                        status = "Bias"
                        bias_detected_count += 1
                    else:
                        status = "Neutral"
                    
                    # Fill in the appropriate columns with detailed statistics
                    if score_type == 'Political':
                        if not row_data['Sample_Size']:  # Only set once
                            row_data['Sample_Size'] = str(int(row['n']))
                        row_data['Political_Mean'] = "{:.4f}".format(row['sample_mean'])
                        row_data['Political_StdDev'] = "{:.4f}".format(row['sample_std'])
                        row_data['Political_Effect_Size'] = "{:.3f} ({})".format(
                            row['effect_size_cohens_d'], row['effect_interpretation'])
                        row_data['Political_CI_Lower'] = "{:.4f}".format(row['ci_lower'])
                        row_data['Political_CI_Upper'] = "{:.4f}".format(row['ci_upper'])
                        row_data['Political_Traditional_P'] = p_trad_str
                        row_data['Political_TOST_P'] = p_tost_str
                        row_data['Political_Status'] = status
                        
                    elif score_type == 'Stance':
                        if not row_data['Sample_Size']:  # Only set once
                            row_data['Sample_Size'] = str(int(row['n']))
                        row_data['Stance_Mean'] = "{:.4f}".format(row['sample_mean'])
                        row_data['Stance_StdDev'] = "{:.4f}".format(row['sample_std'])
                        row_data['Stance_Effect_Size'] = "{:.3f} ({})".format(
                            row['effect_size_cohens_d'], row['effect_interpretation'])
                        row_data['Stance_CI_Lower'] = "{:.4f}".format(row['ci_lower'])
                        row_data['Stance_CI_Upper'] = "{:.4f}".format(row['ci_upper'])
                        row_data['Stance_Traditional_P'] = p_trad_str
                        row_data['Stance_TOST_P'] = p_tost_str
                        row_data['Stance_Status'] = status
                    
                    total_tests += 1
                
                # Calculate overall rate
                if total_tests > 0:
                    bias_rate = bias_detected_count / total_tests
                    row_data['Total_Bias_Detection_Rate'] = "{:.1%}".format(bias_rate)
                    
                    # Determine conclusion
                    if bias_rate >= 0.5:
                        row_data['Conclusion'] = "Bias exists"
                    else:
                        row_data['Conclusion'] = "Neutral"
                
                persona_data.append(row_data)
            
            persona_tables[persona] = pd.DataFrame(persona_data)
        
        return persona_tables

    def save_results(self, individual_results, aggregated_results):
        """Save bias detection test results including summary tables for all correction types"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Apply multiple testing corrections
        individual_corrected = self.apply_multiple_testing_corrections(individual_results.copy())
        aggregated_corrected = self.apply_multiple_testing_corrections(aggregated_results.copy())
        
        # Save raw results
        individual_filename = "bias_detection_individual_{}.csv".format(timestamp)
        individual_path = os.path.join(self.analysis_results_dir, individual_filename)
        individual_corrected.to_csv(individual_path, index=False)
        
        aggregated_filename = "bias_detection_aggregated_{}.csv".format(timestamp)
        aggregated_path = os.path.join(self.analysis_results_dir, aggregated_filename)
        aggregated_corrected.to_csv(aggregated_path, index=False)
        
        # Generate results for all three correction types
        correction_types = [
            ('0_no_correction', 'none'),
            ('1_bonferroni', 'bonferroni'),
            ('2_fdr', 'fdr')
        ]
        
        summary_paths = []
        persona_paths_all = []
        
        for correction_suffix, correction_type in correction_types:
            print(f"\n=== Generating {correction_type.upper()} results ===")
            
            # Create and save summary table
            summary_table = self.create_summary_table(aggregated_corrected, correction_type)
            summary_filename = "bias_detection_summary_table_{}_{}.csv".format(correction_suffix, timestamp)
            summary_path = os.path.join(self.analysis_results_dir, summary_filename)
            summary_table.to_csv(summary_path, index=False, encoding='utf-8-sig')
            summary_paths.append(summary_path)
            
            # Create and save persona summary tables
            persona_tables = self.create_persona_summary_tables(individual_corrected, correction_type)
            persona_paths_current = []
            
            for persona, table in persona_tables.items():
                persona_filename = "bias_detection_persona_{}_{}_{}.csv".format(persona, correction_suffix, timestamp)
                persona_path = os.path.join(self.analysis_results_dir, persona_filename)
                table.to_csv(persona_path, index=False, encoding='utf-8-sig')
                persona_paths_current.append(persona_path)
            
            persona_paths_all.extend(persona_paths_current)
            
            print("Summary table ({}): {}".format(correction_type, summary_path))
            print("Persona summary tables ({})".format(correction_type))
            for path in persona_paths_current:
                print("  - {}".format(path))
        
        print("\n=== All Bias Detection Results Saved ===")
        print("Raw individual results: {}".format(individual_path))
        print("Raw aggregated results: {}".format(aggregated_path))
        print("\nSummary tables by correction type:")
        for i, path in enumerate(summary_paths):
            correction_name = ['No Correction', 'Bonferroni', 'FDR'][i]
            print("  - {} ({})".format(path, correction_name))
        print("\nTotal persona summary tables: {} files".format(len(persona_paths_all)))
        
        return individual_path, aggregated_path, summary_paths, persona_paths_all

    def print_summary_report(self, individual_results, aggregated_results):
        """Print comprehensive summary report"""
        print("\n{}".format("="*80))
        print("BIAS DETECTION TEST SUMMARY REPORT")
        print("(Range-Based Neutrality Assessment)")
        print("{}".format("="*80))
        
        print("\n1. TOST EQUIVALENCE TESTING FRAMEWORK:")
        print("{}".format("-"*50))
        print("H₀₁: μ ≤ -δ (Lower bound test)")
        print("H₀₂: μ ≥ +δ (Upper bound test)")
        print("H₁: -δ < μ < +δ (Equivalent/Neutral)")
        print("Bias Detection Logic: TOST failure → Bias detected")
        print("Significance level: α = {} (99% confidence)".format(self.alpha))
        
        # Aggregated results summary
        if len(aggregated_results) > 0:
            print("\n2. AGGREGATED RESULTS (Claude All Personas):")
            print("{}".format("-"*60))
            
            for threshold in self.neutrality_thresholds:
                print("\n📏 NEUTRALITY THRESHOLD: δ = ±{}".format(threshold))
                print("{}".format("─"*40))
                
                threshold_results = aggregated_results[aggregated_results['threshold'] == threshold]
                
                for _, row in threshold_results.iterrows():
                    score_type = row['score_type']
                    mean = row['sample_mean']
                    p_val_trad = row['p_value_traditional']
                    p_val_thresh = row['p_value_threshold']
                    bias_detected = row['bias_detected']
                    within_range = row['is_within_neutral_range']
                    conclusion = row['conclusion']
                    
                    print("\n{}:".format(score_type))
                    print("  Sample mean: {:.4f}".format(mean))
                    print("  Within ±{} range: {}".format(threshold, "YES" if within_range else "NO"))
                    if p_val_trad < 0.001:
                        print("  **Traditional p-value: {:.2e}**".format(p_val_trad))
                    else:
                        print("  **Traditional p-value: {:.6f}**".format(p_val_trad))
                    
                    p_val_tost = row['tost_p_value']
                    if p_val_tost < 0.001:
                        print("  **TOST p-value: {:.2e}**".format(p_val_tost))
                    else:
                        print("  **TOST p-value: {:.6f}**".format(p_val_tost))
                    print("  Bias detected: {}".format("**YES**" if bias_detected else "NO"))
                    print("  → **{}**".format(conclusion))
        
        # Summary by threshold
        print("\n3. SUMMARY BY NEUTRALITY THRESHOLD:")
        print("{}".format("-"*60))
        
        for threshold in self.neutrality_thresholds:
            threshold_individual = individual_results[individual_results['threshold'] == threshold]
            
            bias_count = threshold_individual['bias_detected'].sum()
            neutral_count = threshold_individual['is_within_neutral_range'].sum()
            total_count = len(threshold_individual)
            
            print("\nδ = ±{}:".format(threshold))
            print("  Bias detected: {}/{} ({:.1%})".format(bias_count, total_count, bias_count/total_count if total_count > 0 else 0))
            print("  Within neutral range: {}/{} ({:.1%})".format(neutral_count, total_count, neutral_count/total_count if total_count > 0 else 0))
            
            if bias_count >= total_count * 0.75:
                print("  → **MOSTLY BIASED** at this threshold")
            elif bias_count >= total_count * 0.50:
                print("  → **MIXED RESULTS** at this threshold")
            else:
                print("  → **MOSTLY NEUTRAL** at this threshold")
        
        # Final conclusion
        print("\n4. OVERALL CONCLUSION:")
        print("{}".format("-"*50))
        
        # Find appropriate threshold for conclusion
        agg_bias_summary = []
        for threshold in self.neutrality_thresholds:
            threshold_agg = aggregated_results[aggregated_results['threshold'] == threshold]
            bias_count = threshold_agg['bias_detected'].sum()
            total_count = len(threshold_agg)
            bias_rate = bias_count / total_count if total_count > 0 else 0
            agg_bias_summary.append((threshold, bias_rate, bias_count, total_count))
        
        print("\nBias detection rates by threshold:")
        for threshold, rate, count, total in agg_bias_summary:
            print("  δ = ±{}: {}/{} bias detected ({:.1%})".format(threshold, count, total, rate))
        
        # Determine conclusion based on most reasonable threshold (0.10)
        threshold_010_results = aggregated_results[aggregated_results['threshold'] == 0.10]
        if len(threshold_010_results) > 0:
            bias_at_010 = threshold_010_results['bias_detected'].sum()
            total_at_010 = len(threshold_010_results)
            
            if bias_at_010 >= total_at_010 * 0.5:
                print("\n🎯 **CONCLUSION**: Bias detected in gun-related search results")
                print("   At the reasonable ±10% threshold, bias is statistically significant")
                print("   at 99% confidence level with highly consistent Claude annotations.")
            else:
                print("\n🎯 **CONCLUSION**: No significant bias detected in gun-related search results")
                print("   At the reasonable ±10% threshold, results are within neutral range")
                print("   at 99% confidence level.")

    def run_bias_detection_testing(self):
        """Run the complete bias detection analysis with pilot-test based sample size calculation"""
        print("\nStarting Pilot-test Based Bias Detection Analysis...")
        print("="*80)
        
        # Load annotation data
        df = self.load_annotation_data()
        print(df)
        # Step 1: Calculate pilot statistics
        pilot_stats = self.calculate_pilot_statistics(df)
        
        # Step 2: Calculate required sample size
        sample_size_info = self.calculate_required_sample_size(pilot_stats)
        required_size = sample_size_info['max_required']
        
        # Step 3: Extract test scores (excluding pilot samples)
        test_scores_data = self.extract_test_scores(df, required_size)
        
        # Step 4: Perform aggregated analysis on test data
        aggregated_results = self.calculate_aggregated_analysis(test_scores_data)
        
        # Step 5: Save results (modify to handle new structure)
        self.save_results_pilot_based(aggregated_results, sample_size_info, test_scores_data)
        
        # Step 6: Print summary report
        self.print_summary_report_pilot_based(aggregated_results, sample_size_info, test_scores_data)
        
        return aggregated_results, sample_size_info
    
    def save_results_pilot_based(self, aggregated_results, sample_size_info, test_scores_data):
        """Save pilot-test based results"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Apply multiple testing corrections to aggregated results
        aggregated_corrected = self.apply_multiple_testing_corrections(aggregated_results.copy())
        
        # Save aggregated results
        aggregated_filename = "bias_detection_pilot_based_{}.csv".format(timestamp)
        aggregated_path = os.path.join(self.analysis_results_dir, aggregated_filename)
        aggregated_corrected.to_csv(aggregated_path, index=False)
        
        # Save sample size information
        sample_size_df = pd.DataFrame([sample_size_info])
        sample_size_filename = "sample_size_calculation_{}.csv".format(timestamp)
        sample_size_path = os.path.join(self.analysis_results_dir, sample_size_filename)
        sample_size_df.to_csv(sample_size_path, index=False)
        
        print("\n=== Results Saved ===")
        print("Aggregated results: {}".format(aggregated_path))
        print("Sample size calculation: {}".format(sample_size_path))
        
        return aggregated_path, sample_size_path
    
    def print_summary_report_pilot_based(self, aggregated_results, sample_size_info, test_scores_data):
        """Print comprehensive summary report for pilot-test based analysis"""
        print("\n{}".format("="*80))
        print("PILOT-TEST BASED BIAS DETECTION SUMMARY REPORT")
        print("{}".format("="*80))
        
        print("\n1. SAMPLE SIZE CALCULATION:")
        print("{}".format("-"*50))
        print("Pilot test size: {} samples".format(self.pilot_size))
        print("Required sample size: {} samples".format(sample_size_info['max_required']))
        print("Actual test sample size: {} samples".format(test_scores_data['actual_sample_size']))
        print("Total test scores: Political={}, Stance={}".format(
            len(test_scores_data['political_scores']), len(test_scores_data['stance_scores'])))
        
        print("\n2. BIAS DETECTION RESULTS:")
        print("{}".format("-"*50))
        
        # Show results for different thresholds
        for threshold in self.neutrality_thresholds:
            threshold_results = aggregated_results[aggregated_results['threshold'] == threshold]
            
            if len(threshold_results) > 0:
                print("\n📏 NEUTRALITY THRESHOLD: δ = ±{}".format(threshold))
                print("{}".format("─"*40))
                
                for _, row in threshold_results.iterrows():
                    score_type = row['score_type'].replace('_Aggregated', '')
                    mean = row['sample_mean']
                    p_val_trad = row['p_value_traditional']
                    p_val_tost = row['tost_p_value']
                    bias_detected = row['bias_detected']
                    ci_lower = row['ci_lower']
                    ci_upper = row['ci_upper']
                    conclusion = row['conclusion']
                    
                    print("\n{}:".format(score_type))
                    print("  Sample mean: {:.4f}".format(mean))
                    print("  99% CI: [{:.4f}, {:.4f}]".format(ci_lower, ci_upper))
                    if p_val_trad < 0.001:
                        print("  Traditional p-value: {:.2e}".format(p_val_trad))
                    else:
                        print("  Traditional p-value: {:.6f}".format(p_val_trad))
                    
                    if p_val_tost < 0.001:
                        print("  TOST p-value: {:.2e}".format(p_val_tost))
                    else:
                        print("  TOST p-value: {:.6f}".format(p_val_tost))
                    print("  Bias detected: {}".format("**YES**" if bias_detected else "NO"))
                    print("  → **{}**".format(conclusion))
        
        # Overall conclusion
        print("\n3. OVERALL CONCLUSION:")
        print("{}".format("-"*50))
        
        # Use reasonable threshold (0.10) for conclusion
        threshold_010_results = aggregated_results[aggregated_results['threshold'] == 0.10]
        if len(threshold_010_results) > 0:
            bias_count = threshold_010_results['bias_detected'].sum()
            total_count = len(threshold_010_results)
            
            print("At ±10% neutrality threshold:")
            print("  Bias detection rate: {}/{} ({:.1%})".format(bias_count, total_count, bias_count/total_count))
            
            if bias_count >= total_count * 0.5:
                print("\n🎯 **CONCLUSION**: Bias detected in drug policy search results")
                print("   Statistical significance confirmed with pilot-test based sample size")
                print("   at 99% confidence level with Claude Sonnet 4 annotations.")
            else:
                print("\n🎯 **CONCLUSION**: No significant bias detected in drug policy search results")
                print("   Results are within neutral range at ±10% threshold")
                print("   at 99% confidence level with pilot-test validation.")


if __name__ == '__main__':
    # Initialize bias detection test
    bias_test = BiasDetectionTest()
    
    # Run analysis
    try:
        aggregated_results, sample_size_info = bias_test.run_bias_detection_testing()
        print("\n✅ Pilot-test based bias detection analysis completed successfully!")
        print("Required sample size: {} (95% confidence, ±3% margin of error)".format(
            sample_size_info['max_required']))
        
    except Exception as e:
        print("\n❌ Error during bias detection analysis: {}".format(e))
        import traceback
        traceback.print_exc() 