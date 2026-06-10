import pandas as pd
import joblib
import os

class DataProcessor:
    def __init__(self, features_path='models/model_features.pkl'):
        """
        Initializes the processor by loading the exact feature list 
        required by the production model.
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature requirements file not found at {features_path}")
        
        self.required_features = joblib.load(features_path)

    def standardize_columns(self, input_df):
        """
        Maps standard SaaS vocabulary columns to the model's required internal feature names.
        Also handles unit conversions (e.g., minutes to seconds for usage duration).
        """
        df = input_df.copy()
        
        # Mapping table for columns (case-insensitive and standard space/underscore replacements)
        mapping = {
            'customer id': 'Customer ID',
            'customer_id': 'Customer ID',
            'account id': 'Customer ID',
            'account_id': 'Customer ID',
            
            'total_transactions': 'total_transactions',
            'billing cycles': 'total_transactions',
            'billing_cycles': 'total_transactions',
            'transactions': 'total_transactions',
            
            'total_revenue': 'total_revenue',
            'ltv ($)': 'total_revenue',
            'ltv': 'total_revenue',
            'lifetime value': 'total_revenue',
            'lifetime_value': 'total_revenue',
            
            'currently_auto_renews': 'currently_auto_renews',
            'auto-renew status': 'currently_auto_renews',
            'auto_renew_status': 'currently_auto_renews',
            'auto-renew': 'currently_auto_renews',
            'auto_renew': 'currently_auto_renews',
            'auto renew status': 'currently_auto_renews',
            'auto renew': 'currently_auto_renews',
            
            'has_cancelled_before': 'has_cancelled_before',
            'prior cancellations': 'has_cancelled_before',
            'prior_cancellations': 'has_cancelled_before',
            'cancelled before': 'has_cancelled_before',
            'cancelled_before': 'has_cancelled_before',
            
            'total_active_days': 'total_active_days',
            'active sessions (30-day)': 'total_active_days',
            'active_sessions_(30-day)': 'total_active_days',
            'active sessions': 'total_active_days',
            'active_sessions': 'total_active_days',
            
            'total_songs_skipped': 'total_songs_skipped',
            'failed sessions / errors': 'total_songs_skipped',
            'failed_sessions_/_errors': 'total_songs_skipped',
            'failed sessions': 'total_songs_skipped',
            'failed_sessions': 'total_songs_skipped',
            'feature drops': 'total_songs_skipped',
            'feature_drops': 'total_songs_skipped',
            'features skipped/drops': 'total_songs_skipped',
            
            'total_songs_completed': 'total_songs_completed',
            'core feature adoptions': 'total_songs_completed',
            'core_feature_adoptions': 'total_songs_completed',
            'core features used': 'total_songs_completed',
            'core_features_used': 'total_songs_completed',
            'features completed': 'total_songs_completed',
            'features_completed': 'total_songs_completed',
            
            'total_listen_time_secs': 'total_listen_time_secs',
            'product usage duration (mins)': 'total_listen_time_secs',
            'product_usage_duration_(mins)': 'total_listen_time_secs',
            'product usage duration': 'total_listen_time_secs',
            'product_usage_duration': 'total_listen_time_secs',
            'usage duration (mins)': 'total_listen_time_secs',
            'usage_duration_(mins)': 'total_listen_time_secs',
            'usage duration': 'total_listen_time_secs',
        }
        
        # Clean current columns to lowercase strip to make matching robust
        col_map = {}
        for col in df.columns:
            cleaned_col = str(col).strip().lower()
            if cleaned_col in mapping:
                col_map[col] = mapping[cleaned_col]
        
        # Rename columns using matching mapping
        df = df.rename(columns=col_map)
        
        # Unit conversion check for duration (minutes to seconds)
        for orig_col, mapped_col in col_map.items():
            if mapped_col == 'total_listen_time_secs':
                orig_cleaned = str(orig_col).strip().lower()
                # If the column name explicitly says mins/minutes or we detect the usage duration was in minutes
                if 'mins' in orig_cleaned or 'minute' in orig_cleaned or 'duration' in orig_cleaned:
                    if 'secs' not in orig_cleaned and 'second' not in orig_cleaned:
                        df['total_listen_time_secs'] = df['total_listen_time_secs'] * 60.0
                        
        return df

    def prepare_inference_data(self, input_data):
        """
        Cleans, standardizes, and formats incoming client data for the LightGBM model.
        
        Args:
            input_data (pd.DataFrame): Raw data uploaded by the client.
            
        Returns:
            pd.DataFrame: A fully processed dataframe ready for model prediction.
        """
        # Run standardization to map SaaS terms to model columns
        df = self.standardize_columns(input_data)

        # Fill missing behavioral/transactional data with 0 (as established in Phase 2)
        fill_values = {
            'total_transactions': 0, 'total_revenue': 0, 
            'currently_auto_renews': 0, 'has_cancelled_before': 0,
            'total_active_days': 0, 'total_songs_skipped': 0,
            'total_songs_completed': 0, 'total_listen_time_secs': 0
        }
        df = df.fillna(value=fill_values)

        # Ensure all required features are present, fill with 0 if completely missing
        for col in self.required_features:
            if col not in df.columns:
                df[col] = 0
                
        # Reorder columns to strictly match the model's expected input matrix
        df = df[self.required_features]
        
        return df

    def calibrate_probability(self, row, raw_prob):
        """
        Applies business rule guardrails to calibrate prediction probabilities,
        preventing false positives for highly active power users with active subscriptions.
        """
        prob = raw_prob
        
        # Standardize row keys for comparison
        clean_row = {}
        for k, v in row.items():
            # Standardize keys by mapping SaaS terms
            mapping = {
                'customer id': 'Customer ID',
                'customer_id': 'Customer ID',
                'account id': 'Customer ID',
                'account_id': 'Customer ID',
                
                'total_transactions': 'total_transactions',
                'billing cycles': 'total_transactions',
                'billing_cycles': 'total_transactions',
                'transactions': 'total_transactions',
                
                'total_revenue': 'total_revenue',
                'ltv ($)': 'total_revenue',
                'ltv': 'total_revenue',
                'lifetime value': 'total_revenue',
                'lifetime_value': 'total_revenue',
                
                'currently_auto_renews': 'currently_auto_renews',
                'auto-renew status': 'currently_auto_renews',
                'auto_renew_status': 'currently_auto_renews',
                'auto-renew': 'currently_auto_renews',
                'auto_renew': 'currently_auto_renews',
                'auto renew status': 'currently_auto_renews',
                'auto renew': 'currently_auto_renews',
                
                'has_cancelled_before': 'has_cancelled_before',
                'prior cancellations': 'has_cancelled_before',
                'prior_cancellations': 'has_cancelled_before',
                'cancelled before': 'has_cancelled_before',
                'cancelled_before': 'has_cancelled_before',
                
                'total_active_days': 'total_active_days',
                'active sessions (30-day)': 'total_active_days',
                'active_sessions_(30-day)': 'total_active_days',
                'active sessions': 'total_active_days',
                'active_sessions': 'total_active_days',
                
                'total_songs_skipped': 'total_songs_skipped',
                'failed sessions / errors': 'total_songs_skipped',
                'failed_sessions_/_errors': 'total_songs_skipped',
                'failed sessions': 'total_songs_skipped',
                'failed_sessions': 'total_songs_skipped',
                'feature drops': 'total_songs_skipped',
                'feature_drops': 'total_songs_skipped',
                'features skipped/drops': 'total_songs_skipped',
                
                'total_songs_completed': 'total_songs_completed',
                'core feature adoptions': 'total_songs_completed',
                'core_feature_adoptions': 'total_songs_completed',
                'core features used': 'total_songs_completed',
                'core_features_used': 'total_songs_completed',
                'features completed': 'total_songs_completed',
                'features_completed': 'total_songs_completed',
                
                'total_listen_time_secs': 'total_listen_time_secs',
                'product usage duration (mins)': 'total_listen_time_secs',
                'product_usage_duration_(mins)': 'total_listen_time_secs',
                'product usage duration': 'total_listen_time_secs',
                'product_usage_duration': 'total_listen_time_secs',
                'usage duration (mins)': 'total_listen_time_secs',
                'usage_duration_(mins)': 'total_listen_time_secs',
                'usage duration': 'total_listen_time_secs',
            }
            cleaned_key = str(k).strip().lower()
            if cleaned_key in mapping:
                clean_row[mapping[cleaned_key]] = v
            else:
                clean_row[k] = v

        active_days = int(clean_row.get('total_active_days', 0))
        auto_renew = int(clean_row.get('currently_auto_renews', 0))
        completed = int(clean_row.get('total_songs_completed', 0))
        skipped = int(clean_row.get('total_songs_skipped', 0))
        total_actions = completed + skipped
        skip_ratio = skipped / max(1, total_actions)
        
        # Rule 1: Highly active champion user with active auto-renew
        if auto_renew == 1 and active_days >= 20:
            if completed >= 100 and skip_ratio <= 0.15:
                prob = min(prob, 15.0)
            elif completed >= 50 and skip_ratio <= 0.25:
                prob = min(prob, 35.0)
                
        # Rule 2: Completely inactive accounts with no auto-renew
        if active_days <= 1 and auto_renew == 0:
            prob = max(prob, 95.0)
            
        # Rule 3: Zombie Users (Auto-Renew ON but zero activity in 30 days)
        # In SaaS, paying but not using is a major retention threat.
        if active_days == 0 and auto_renew == 1:
            prob = max(prob, 45.0)
            
        # Rule 4: Low usage with auto-renew active
        # User logged in <= 3 times. They are not adopted.
        if 0 < active_days <= 3 and auto_renew == 1:
            prob = max(prob, 35.0)
            
        # Rule 5: Low usage with auto-renew disabled
        if active_days <= 5 and auto_renew == 0:
            prob = max(prob, 75.0)
            
        # Rule 6: Active user (active_days >= 15) with auto-renew active should not be High Flight Risk
        if auto_renew == 1 and active_days >= 15:
            if skip_ratio <= 0.30:
                prob = min(prob, 40.0)  # Healthy or low-level At Risk
            else:
                prob = min(prob, 55.0)  # At Risk, but not High Flight Risk
                
        # Rule 7: Champion user (active_days >= 25) with auto-renew active and low errors is definitely healthy.
        if auto_renew == 1 and active_days >= 25 and skip_ratio <= 0.20:
            prob = min(prob, 25.0)
            
        return prob