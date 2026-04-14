import os
import pandas as pd
import json
import numpy as np
import math
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

base_filename = "annotated_gun_control_20250731_231907"
sample_size = 10000

class SampleSizeCalculator:
    def __init__(self, base_filename=base_filename):
        self.base_filename = base_filename
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.annotation_dir = os.path.join(self.current_dir, '../annotation_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        
        # Create analysis results directory if it doesn't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        
        # Claude 모델만 사용 (일관성이 98-99%로 높음, 메인 통계 검정과 일치)
        # claude-sonnet-4-20250514
        self.models = ['claude-sonnet-4-20250514', 'gpt-4.1']  # GPT-4o 제외
        # self.models = ['claude-sonnet-4-20250514']
        self.personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
        self.model_persona_combinations = []
        
        for model in self.models:
            for persona in self.personas:
                self.model_persona_combinations.append(f"{model}_{persona}")
        
        print(f"Sample size calculation for Claude model only")
        print(f"Model-persona combinations: {len(self.model_persona_combinations)}")
        for combo in self.model_persona_combinations:
            print(f"  - {combo}")
        print(f"Note: Using Claude-only for consistency with main statistical test")

    def load_annotation_data(self, num_files=3):
        """Load annotation data from multiple repetition files"""
        all_data = []
        
        print(f"\nLoading data from {num_files} annotation files...")
        filename = f"{self.base_filename}.csv"
        filepath = os.path.join(self.annotation_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = df[:sample_size] 
            all_data.append(df)
            print(f"✓ Loaded {filename} - Shape: {df.shape}")
        else:
            print(f"✗ File not found: {filename}")
        
        if not all_data:
            raise ValueError("No annotation files found!")
        
        # Combine all repetitions
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df

    def parse_json_response(self, json_str):
        """Parse JSON response and extract scores"""
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

    def extract_all_scores(self, df):
        """Extract all valid scores from the dataset using row-wise averages"""
        print("\nExtracting scores from all annotations...")
        print("Using row-wise averaging (4 personas per row)")
        
        political_row_averages = []
        stance_row_averages = []
        
        valid_rows = 0
        
        for idx, row in df.iterrows():
            # Extract scores for all 4 personas for this row
            row_political_scores = []
            row_stance_scores = []
            
            for combo in self.model_persona_combinations:
                if combo in df.columns:
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
                
            if (len(row_political_scores) == len(self.model_persona_combinations) and 
                len(row_stance_scores) == len(self.model_persona_combinations)):
                valid_rows += 1
        
        scores_data = {
            'row_averaged': {
                'political_scores': political_row_averages,
                'stance_scores': stance_row_averages,
                'political_count': len(political_row_averages),
                'stance_count': len(stance_row_averages)
            }
        }
        
        print(f"  Total rows: {self.model_persona_combinations} ")
        print(f"  Valid rows with all 4 personas: {valid_rows}")
        print(f"  Political row averages: {len(political_row_averages)}")
        print(f"  Stance row averages: {len(stance_row_averages)}")
        print(f"  Note: Each row average represents mean of 4 persona scores")
        
        return scores_data

    def calculate_sample_statistics(self, scores_data):
        """Calculate sample statistics for row-averaged scores"""
        print("\nCalculating sample statistics...")
        print("Statistics based on row-averaged scores (each row = average of 4 personas)")
        
        statistics = []
        
        # We now have only one entry: 'row_averaged'
        data = scores_data['row_averaged']
        pol_scores = data['political_scores']
        stance_scores = data['stance_scores']
        
        if len(pol_scores) > 0:
            pol_mean = np.mean(pol_scores)
            pol_std = np.std(pol_scores, ddof=1)  # Sample standard deviation
            pol_var = np.var(pol_scores, ddof=1)  # Sample variance
            pol_min = np.min(pol_scores)
            pol_max = np.max(pol_scores)
            pol_range = pol_max - pol_min
        else:
            pol_mean = pol_std = pol_var = pol_min = pol_max = pol_range = None
        
        if len(stance_scores) > 0:
            stance_mean = np.mean(stance_scores)
            stance_std = np.std(stance_scores, ddof=1)  # Sample standard deviation
            stance_var = np.var(stance_scores, ddof=1)  # Sample variance
            stance_min = np.min(stance_scores)
            stance_max = np.max(stance_scores)
            stance_range = stance_max - stance_min
        else:
            stance_mean = stance_std = stance_var = stance_min = stance_max = stance_range = None
        
        statistics.append({
            'model_persona': 'claude-sonnet-4-all-personas-averaged',
            'model': 'claude-sonnet-4-20250514',
            'persona': 'all_averaged',
            
            # Political statistics
            'political_sample_size': len(pol_scores),
            'political_mean': pol_mean,
            'political_std': pol_std,
            'political_var': pol_var,
            'political_min': pol_min,
            'political_max': pol_max,
            'political_range': pol_range,
            
            # Stance statistics
            'stance_sample_size': len(stance_scores),
            'stance_mean': stance_mean,
            'stance_std': stance_std,
            'stance_var': stance_var,
            'stance_min': stance_min,
            'stance_max': stance_max,
            'stance_range': stance_range
        })
        
        print(f"Row-averaged statistics:")
        print(f"  Political: n={len(pol_scores)}, mean={pol_mean:.4f}, std={pol_std:.4f}")
        print(f"  Stance: n={len(stance_scores)}, mean={stance_mean:.4f}, std={stance_std:.4f}")
        print(f"  Note: n represents number of news articles (rows), not individual persona ratings")
        
        return pd.DataFrame(statistics)

    def calculate_required_sample_size(self, stats_df, confidence_levels=[0.95], 
                                     margin_errors=[0.03]):
        """
        Calculate required sample size using the formula: n = (Z * σ / E)²
        
        Parameters:
        - confidence_levels: List of confidence levels (e.g., 0.95 for 95%)
        - margin_errors: List of margin of errors (e.g., 0.05 for 5% of the scale)
        """
        print("\nCalculating required sample sizes...")
        
        # Z-scores for different confidence levels
        z_scores = {
            # 0.90: 1.645,  # 90% confidence
            0.95: 1.960,  # 95% confidence
            # 0.99: 2.576   # 99% confidence
        }
        
        sample_size_results = []
        
        for _, row in stats_df.iterrows():
            for conf_level in confidence_levels:
                z_score = z_scores[conf_level]
                
                for margin_error in margin_errors:
                    # Political sample size calculation
                    if row['political_std'] is not None and row['political_std'] > 0:
                        pol_sample_size = math.ceil((z_score * row['political_std'] / margin_error) ** 2)
                    else:
                        pol_sample_size = None
                    
                    # Stance sample size calculation
                    if row['stance_std'] is not None and row['stance_std'] > 0:
                        stance_sample_size = math.ceil((z_score * row['stance_std'] / margin_error) ** 2)
                    else:
                        stance_sample_size = None
                    
                    sample_size_results.append({
                        'model_persona': row['model_persona'],
                        'confidence_level': conf_level,
                        'margin_error': margin_error,
                        'z_score': z_score,
                        
                        'current_political_sample_size': row['political_sample_size'],
                        'required_political_sample_size': pol_sample_size,
                        'political_std_used': row['political_std'],
                        
                        'current_stance_sample_size': row['stance_sample_size'],
                        'required_stance_sample_size': stance_sample_size,
                        'stance_std_used': row['stance_std'],
                        
                        'max_required_sample_size': max(pol_sample_size or 0, stance_sample_size or 0) 
                                                  if pol_sample_size is not None or stance_sample_size is not None else None
                    })
        
        return pd.DataFrame(sample_size_results)

    def calculate_overall_recommendations(self, sample_size_df):
        """Calculate overall sample size recommendations"""
        print("\nCalculating overall recommendations...")
        
        recommendations = []
        
        # Group by confidence level and margin error
        for conf_level in sample_size_df['confidence_level'].unique():
            for margin_error in sample_size_df['margin_error'].unique():
                subset = sample_size_df[
                    (sample_size_df['confidence_level'] == conf_level) & 
                    (sample_size_df['margin_error'] == margin_error)
                ]
                
                # Find maximum required sample size across all model-persona combinations
                max_pol_size = subset['required_political_sample_size'].max()
                max_stance_size = subset['required_stance_sample_size'].max()
                max_overall_size = subset['max_required_sample_size'].max()
                
                # Current sample size (should be same across all combinations)
                current_size = subset['current_political_sample_size'].iloc[0] if len(subset) > 0 else None
                
                # Additional samples needed
                additional_needed = max(0, max_overall_size - current_size) if max_overall_size and current_size else None
                
                recommendations.append({
                    'confidence_level': conf_level,
                    'margin_error': margin_error,
                    'current_sample_size': current_size,
                    'max_required_political': max_pol_size,
                    'max_required_stance': max_stance_size,
                    'max_required_overall': max_overall_size,
                    'additional_samples_needed': additional_needed,
                    'sampling_adequacy': 'Adequate' if additional_needed == 0 else 'Insufficient'
                })
        
        return pd.DataFrame(recommendations)

    def save_results(self, stats_df, sample_size_df, recommendations_df):
        """Save all results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sample statistics
        stats_filename = f"sample_statistics_{timestamp}.csv"
        stats_path = os.path.join(self.analysis_results_dir, stats_filename)
        stats_df.to_csv(stats_path, index=False)
        
        # Save detailed sample size calculations
        detailed_filename = f"sample_size_detailed_{timestamp}.csv"
        detailed_path = os.path.join(self.analysis_results_dir, detailed_filename)
        sample_size_df.to_csv(detailed_path, index=False)
        
        # Save recommendations
        rec_filename = f"sample_size_recommendations_{timestamp}.csv"
        rec_path = os.path.join(self.analysis_results_dir, rec_filename)
        recommendations_df.to_csv(rec_path, index=False)
        
        print(f"\n=== Results Saved ===")
        print(f"Sample statistics: {stats_path}")
        print(f"Detailed calculations: {detailed_path}")
        print(f"Recommendations: {rec_path}")
        
        return stats_path, detailed_path, rec_path

    def print_summary_report(self, stats_df, recommendations_df):
        """Print a comprehensive summary report"""
        print(f"\n{'='*80}")
        print(f"SAMPLE SIZE ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Current sample statistics summary
        print(f"\n1. CURRENT SAMPLE STATISTICS:")
        print(f"{'-'*50}")
        
        # Overall statistics across all model-persona combinations
        all_pol_scores = []
        all_stance_scores = []
        
        for _, row in stats_df.iterrows():
            if row['political_std'] is not None:
                all_pol_scores.append(row['political_std'])
            if row['stance_std'] is not None:
                all_stance_scores.append(row['stance_std'])
        
        if all_pol_scores:
            print(f"Political Score Standard Deviation:")
            print(f"  Mean: {np.mean(all_pol_scores):.4f}")
            print(f"  Min:  {np.min(all_pol_scores):.4f}")
            print(f"  Max:  {np.max(all_pol_scores):.4f}")
        
        if all_stance_scores:
            print(f"Stance Score Standard Deviation:")
            print(f"  Mean: {np.mean(all_stance_scores):.4f}")
            print(f"  Min:  {np.min(all_stance_scores):.4f}")
            print(f"  Max:  {np.max(all_stance_scores):.4f}")
        
        current_sample = stats_df['political_sample_size'].iloc[0] if len(stats_df) > 0 else 0
        print(f"Current Sample Size: {current_sample}")
        
        # Sample size recommendations
        print(f"\n2. SAMPLE SIZE RECOMMENDATIONS:")
        print(f"{'-'*50}")
        
        # Sort by confidence level and margin error
        sorted_rec = recommendations_df.sort_values(['confidence_level', 'margin_error'])
        
        for _, row in sorted_rec.iterrows():
            conf_pct = int(row['confidence_level'] * 100)
            margin_pct = int(row['margin_error'] * 100)
            
            print(f"\n{conf_pct}% Confidence, ±{margin_pct}% Margin of Error:")
            print(f"  Required Sample Size: {row['max_required_overall']}")
            print(f"  Additional Needed: {row['additional_samples_needed']}")
            print(f"  Status: {row['sampling_adequacy']}")
        
        # Most conservative recommendation
        max_required = recommendations_df['max_required_overall'].max()
        max_additional = recommendations_df['additional_samples_needed'].max()
        
        print(f"\n3. CONSERVATIVE RECOMMENDATION:")
        print(f"{'-'*50}")
        print(f"Most Conservative Required Sample Size: {max_required}")
        print(f"Additional Samples Needed: {max_additional}")
        print(f"This ensures 99% confidence with ±5% margin of error")

    def run_analysis(self, num_files=3):
        """Run the complete sample size analysis"""
        print(f"Starting sample size analysis using {num_files} annotation files...")
        
        # Load data
        combined_df = self.load_annotation_data(num_files)
        
        # Extract scores
        scores_data = self.extract_all_scores(combined_df)
        
        # Calculate sample statistics
        stats_df = self.calculate_sample_statistics(scores_data)
        
        # Calculate required sample sizes
        sample_size_df = self.calculate_required_sample_size(stats_df)
        
        # Calculate recommendations
        recommendations_df = self.calculate_overall_recommendations(sample_size_df)
        
        # Save results
        self.save_results(stats_df, sample_size_df, recommendations_df)
        
        # Print report
        self.print_summary_report(stats_df, recommendations_df)
        
        return stats_df, sample_size_df, recommendations_df


if __name__ == '__main__':
    # Initialize calculator
    calculator = SampleSizeCalculator()
    
    # Run analysis
    try:
        stats_results, detailed_results, recommendations = calculator.run_analysis(num_files=3)
        print(f"\n✅ Sample size analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc() 