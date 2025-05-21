import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # For creating directories
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay


class HotelBookingAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        self.plot_dir = "plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December']
        # Initialize other attributes like self.X_processed, self.y_target, self.processed_feature_names
        self.X_processed = None
        self.y_target = None
        self.processed_feature_names = None
        self.categorical_features = [] # Initialize to avoid AttributeError if feature_engineer not called
        self.numerical_features = []   # Initialize
        self.safe_ohe_features = []

    def load_data(self):
        print(f"Loading data from: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Original shape: {self.df.shape}")
        return self.df

    def clean_data(self):
        if self.df is None:
            self.load_data()

        print("\n--- Starting Data Cleaning ---")
        df_cleaned = self.df.copy()  # Work on a copy

        # 1. Handle 'children'
        df_cleaned['children'] = df_cleaned['children'].fillna(0).astype(int)

        # 2. Handle 'country'
        df_cleaned['country'] = df_cleaned['country'].fillna('Unknown')

        # 3. Handle 'agent'
        df_cleaned['agent'] = df_cleaned['agent'].fillna(0).astype(int)

        # 4. Handle 'company'
        df_cleaned['has_company'] = df_cleaned['company'].notna().astype(int)
        df_cleaned = df_cleaned.drop(columns=['company'])
        print(f"Shape after handling company: {df_cleaned.shape}")

        # 5. Convert 'reservation_status_date'
        df_cleaned['reservation_status_date'] = pd.to_datetime(df_cleaned['reservation_status_date'])

        # 6. Remove negative ADR
        negative_adr_count = len(df_cleaned[df_cleaned['adr'] < 0])
        if negative_adr_count > 0:
            print(f"Found {negative_adr_count} booking(s) with negative ADR. Removing them.")
            df_cleaned = df_cleaned[df_cleaned['adr'] >= 0]
        else:
            print("No bookings with negative ADR found.")

        # 7. Remove zero-guest bookings
        zero_guest_condition = (df_cleaned['adults'] == 0) & (df_cleaned['children'] == 0) & (df_cleaned['babies'] == 0)
        zero_guest_count = len(df_cleaned[zero_guest_condition])
        if zero_guest_count > 0:
            print(f"Found {zero_guest_count} bookings with zero total guests. Removing them.")
            df_cleaned = df_cleaned[~zero_guest_condition]
        else:
            print("No bookings with zero total guests found.")

        print(f"Shape after final cleaning: {df_cleaned.shape}")
        self.cleaned_df = df_cleaned
        print("--- Data Cleaning Complete ---")
        self.cleaned_df.info()
        # Save cleaned data
        cleaned_file_path = 'cleaned_hotel_bookings.csv'
        self.cleaned_df.to_csv(cleaned_file_path, index=False)
        print(f"\nCleaned DataFrame saved to '{cleaned_file_path}'")
        return self.cleaned_df

    def _save_plot(self, fig_title):
        """Helper function to save the current plot."""
        plt.savefig(os.path.join(self.plot_dir, f"{fig_title.replace(' ', '_').lower()}.png"), bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Plot saved: {fig_title}.png")

    def perform_eda(self):
        if self.cleaned_df is None:
            print("Cleaned data not available. Please run clean_data() first.")
            return

        df_eda = self.cleaned_df.copy()  # Use cleaned data for EDA
        print("\n--- Starting Exploratory Data Analysis (EDA) ---")

        # 1. Overall cancellation rate
        cancellation_rate = df_eda['is_canceled'].mean()
        print(f"Overall cancellation rate: {cancellation_rate:.2%}\n")

        # 2. Cancellation rate by hotel type
        plt.figure(figsize=(8, 6))
        sns.countplot(x='hotel', hue='is_canceled', data=df_eda)
        title = 'Cancellation Status by Hotel Type'
        plt.title(title)
        plt.xlabel('Hotel Type')
        plt.ylabel('Number of Bookings')
        plt.legend(title='Canceled', labels=['Not Canceled', 'Canceled'])
        self._save_plot(title)
        print(df_eda.groupby('hotel')['is_canceled'].mean().sort_values(ascending=False))

        # 3. Cancellation rate by Arrival Month

        df_eda['arrival_date_month'] = pd.Categorical(df_eda['arrival_date_month'], categories=self.month_order,
                                                      ordered=True)
        plt.figure(figsize=(12, 7))
        # Add hue for sns.barplot to avoid FutureWarning if needed, though current warning is about palette without hue
        sns.barplot(x='arrival_date_month', y='is_canceled', data=df_eda,
                    estimator=lambda x: sum(x == 1) * 100.0 / len(x),
                    palette='viridis', order=self.month_order, errorbar=None)  # Use errorbar=None
        title = 'Cancellation Rate by Arrival Month'
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Cancellation Rate (%)')
        plt.xticks(rotation=45)
        self._save_plot(title)
        print(df_eda.groupby('arrival_date_month', observed=False)['is_canceled'].mean().sort_values(ascending=False))

        # 4. Lead time vs. cancellations
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df_eda, x='lead_time', hue='is_canceled', kde=True, multiple="stack", bins=50)
        title = 'Distribution of Lead Time by Cancellation Status'
        plt.title(title)
        plt.xlabel('Lead Time (days)')
        plt.ylabel('Number of Bookings')
        plt.legend(title='Canceled', labels=['Canceled', 'Not Canceled'])
        plt.xlim(0, 400)
        self._save_plot(title)

        df_eda['lead_time_bins'] = pd.cut(df_eda['lead_time'],
                                          bins=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 600, 900],
                                          labels=['0-30', '31-60', '61-90', '91-120', '121-150', '151-180', '181-210',
                                                  '211-240', '241-270', '271-300', '301-600', '601+'],
                                          right=False)  # added right=False to include 0
        plt.figure(figsize=(12, 7))
        sns.pointplot(x='lead_time_bins', y='is_canceled', data=df_eda, errorbar=None)  # Use errorbar=None
        title = 'Cancellation Rate by Lead Time Bins'
        plt.title(title)
        plt.xlabel('Lead Time (Days Binned)')
        plt.ylabel('Cancellation Rate')
        plt.xticks(rotation=45)
        self._save_plot(title)
        print(df_eda.groupby('lead_time_bins', observed=False)['is_canceled'].mean().sort_values(ascending=False))

        # 5. ADR vs. cancellations
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df_eda, x='adr', hue='is_canceled', kde=True, multiple="stack", bins=50)
        title = 'Distribution of ADR by Cancellation Status'
        plt.title(title)
        plt.xlabel('Average Daily Rate (ADR)')
        plt.ylabel('Number of Bookings')
        plt.legend(title='Canceled', labels=['Canceled', 'Not Canceled'])
        plt.xlim(0, 500)
        self._save_plot(title)

        # Define bins carefully to avoid issues with values at the edge like 0 ADR
        adr_bins_def = [-1, 50, 100, 150, 200, 250, 300, 400, 500, df_eda['adr'].max()]  # Ensure 0 is included
        adr_labels_def = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-400', '401-500', f'501+']
        df_eda['adr_bins'] = pd.cut(df_eda['adr'], bins=adr_bins_def, labels=adr_labels_def,
                                    right=True)  # right=True is default for pd.cut

        plt.figure(figsize=(12, 7))
        sns.pointplot(x='adr_bins', y='is_canceled', data=df_eda, errorbar=None)  # Use errorbar=None
        title = 'Cancellation Rate by ADR Bins'
        plt.title(title)
        plt.xlabel('ADR (Binned)')
        plt.ylabel('Cancellation Rate')
        plt.xticks(rotation=45)
        self._save_plot(title)
        print(df_eda.groupby('adr_bins', observed=False)['is_canceled'].mean().sort_values(ascending=False))

        # ---- NEW EDA STEPS ----
        print("\n--- Continuing with More EDA ---")

        # 6. Deposit Type
        plt.figure(figsize=(8, 6))
        sns.countplot(x='deposit_type', hue='is_canceled', data=df_eda)
        title = 'Cancellation Status by Deposit Type'
        plt.title(title)
        plt.xlabel('Deposit Type')
        plt.ylabel('Number of Bookings')
        self._save_plot(title)
        print(df_eda.groupby('deposit_type')['is_canceled'].mean().sort_values(ascending=False))

        # 7. Customer Type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='customer_type', hue='is_canceled', data=df_eda)
        title = 'Cancellation Status by Customer Type'
        plt.title(title)
        plt.xlabel('Customer Type')
        plt.ylabel('Number of Bookings')
        self._save_plot(title)
        print(df_eda.groupby('customer_type')['is_canceled'].mean().sort_values(ascending=False))

        # 8. Market Segment
        plt.figure(figsize=(12, 7))
        sns.countplot(x='market_segment', hue='is_canceled', data=df_eda)
        plt.xticks(rotation=45, ha='right')
        title = 'Cancellation Status by Market Segment'
        plt.title(title)
        plt.xlabel('Market Segment')
        plt.ylabel('Number of Bookings')
        self._save_plot(title)
        print(df_eda.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False))

        # 9. Repeated Guest
        plt.figure(figsize=(7, 5))
        sns.countplot(x='is_repeated_guest', hue='is_canceled', data=df_eda)
        title = 'Cancellation Status by Repeated Guest'
        plt.title(title)
        plt.xlabel('Is Repeated Guest (0=No, 1=Yes)')
        plt.ylabel('Number of Bookings')
        self._save_plot(title)
        print(df_eda.groupby('is_repeated_guest')['is_canceled'].mean().sort_values(ascending=False))

        # 10. Correlation Heatmap
        relevant_numeric_cols = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
                                 'adults', 'children', 'babies', 'is_repeated_guest',
                                 'previous_cancellations', 'previous_bookings_not_canceled',
                                 'booking_changes', 'adr', 'required_car_parking_spaces',
                                 'total_of_special_requests', 'is_canceled', 'has_company']

        plt.figure(figsize=(16, 12))  # Increased size for better readability
        correlation_matrix = df_eda[relevant_numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)  # Added linewidths
        title = 'Correlation Heatmap of Relevant Numerical Features'
        plt.title(title)
        self._save_plot(title)
        print("\nCorrelation with 'is_canceled':")
        print(correlation_matrix['is_canceled'].sort_values(ascending=False))

        print("--- EDA Complete ---")

    def feature_engineer(self):
        if self.cleaned_df is None:
            print("Cleaned data not available. Please run clean_data() first.")
            return None

        print("\n--- Starting Feature Engineering ---")
        df_fe = self.cleaned_df.copy()

        # ... (creation of new features like total_stay_duration, total_guests, etc.) ...
        # Example:
        month_to_num = {name: i + 1 for i, name in
                        enumerate(self.month_order)}  # month_order from EDA (ensure it's accessible or defined in class)
        df_fe['arrival_date_month_numeric'] = df_fe['arrival_date_month'].map(month_to_num)
        # ... other feature creations ...

        cols_to_drop_before_model = ['reservation_status', 'reservation_status_date',
                                     'arrival_date_year', 'arrival_date_month',
                                     'arrival_date_week_number', 'arrival_date_day_of_month',
                                     'lead_time_bins', 'adr_bins']

        df_model_data = df_fe.drop(columns=cols_to_drop_before_model, errors='ignore')

        self.categorical_features = df_model_data.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'agent' in df_model_data.columns and 'agent' not in self.categorical_features:
            self.categorical_features.append('agent')

        self.numerical_features = df_model_data.select_dtypes(include=np.number).columns.tolist()
        self.numerical_features = [col for col in self.numerical_features if
                                   col not in ['is_canceled', 'agent'] and col not in self.categorical_features]

        print(f"Identified categorical features: {self.categorical_features}")
        print(f"Identified numerical features: {self.numerical_features}")

        # Define the features to be one-hot encoded explicitly
        self.safe_ohe_features = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'deposit_type',
                                  'customer_type']
        # Ensure these features actually exist in df_model_data.columns to avoid errors
        self.safe_ohe_features = [col for col in self.safe_ohe_features if col in df_model_data.columns]

        print(f"Features for One-Hot Encoding: {self.safe_ohe_features}")

        # Remove ohe_features definition as it's not used.

        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.safe_ohe_features),
                # Use self.safe_ohe_features
                ('scaler', StandardScaler(), self.numerical_features)
            ],
            remainder='passthrough'
        )

        X = df_model_data.drop('is_canceled', axis=1)
        y = df_model_data['is_canceled']

        print("Applying preprocessing transformations...")
        X_processed_np = preprocessor.fit_transform(X)

        ohe_feature_names_out = preprocessor.named_transformers_['onehot'].get_feature_names_out(self.safe_ohe_features)

        # Correctly determine remainder columns based on what wasn't in safe_ohe_features or numerical_features
        processed_cols_in_transformer = self.safe_ohe_features + self.numerical_features
        remainder_cols = [col for col in X.columns if col not in processed_cols_in_transformer]

        self.processed_feature_names = list(ohe_feature_names_out) + self.numerical_features + remainder_cols

        X_processed = pd.DataFrame(X_processed_np, columns=self.processed_feature_names, index=X.index)

        cols_to_drop_after_pass = [col for col in remainder_cols if X_processed[col].dtype == 'object']
        if cols_to_drop_after_pass:
            print(f"Dropping unencoded object columns from passthrough: {cols_to_drop_after_pass}")
            X_processed = X_processed.drop(columns=cols_to_drop_after_pass)
            self.processed_feature_names = [name for name in self.processed_feature_names if
                                            name not in cols_to_drop_after_pass]

        print(f"Shape of processed features (X_processed): {X_processed.shape}")
        print("--- Feature Engineering Complete ---")

        self.X_processed = X_processed
        self.y_target = y

        return self.X_processed, self.y_target

    def build_and_evaluate_model(self):
        if self.X_processed is None or self.y_target is None:
            print("Processed data (X_processed, y_target) not available. Run feature_engineer() first.")
            return

        print("\n--- Starting Model Building and Evaluation ---")

        X = self.X_processed
        y = self.y_target

        # 1. Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y  # Stratify for imbalanced classes
        )
        print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

        # --- Logistic Regression ---
        print("\n--- Logistic Regression ---")
        log_reg = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)  # Added max_iter
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)
        y_pred_proba_log_reg = log_reg.predict_proba(X_test)[:, 1]

        print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
        print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_log_reg))
        print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

        cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
        disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg)
        disp_log_reg.plot()
        title_lr_cm = "Logistic Regression Confusion Matrix"
        plt.title(title_lr_cm)
        self._save_plot(title_lr_cm)  # Assumes _save_plot is part of the class

        # --- Random Forest Classifier ---
        print("\n--- Random Forest Classifier ---")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                        class_weight='balanced')  # Added class_weight
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        y_pred_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

        print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
        print("Random Forest ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))
        print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

        cm_rf = confusion_matrix(y_test, y_pred_rf)
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
        disp_rf.plot()
        title_rf_cm = "Random Forest Confusion Matrix"
        plt.title(title_rf_cm)
        self._save_plot(title_rf_cm)

        # Feature Importances from Random Forest
        if hasattr(rf_clf, 'feature_importances_'):
            importances = rf_clf.feature_importances_
            feature_names = self.processed_feature_names  # Ensure this is set correctly in feature_engineer

            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            print("\nTop 20 Feature Importances (Random Forest):")
            print(feature_importance_df.head(20))

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
            title_fi = "Top 20 Feature Importances (Random Forest)"
            plt.title(title_fi)
            plt.tight_layout()
            self._save_plot(title_fi)

        # Plot ROC Curves
        fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_proba_log_reg)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

        plt.figure(figsize=(10, 7))
        plt.plot(fpr_log_reg, tpr_log_reg,
                 label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_proba_log_reg):.2f})")
        plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba_rf):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title_roc = 'ROC Curves'
        plt.title(title_roc)
        plt.legend()
        self._save_plot(title_roc)

        print("--- Model Building and Evaluation Complete ---")


# How to use the class:
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    analyzer = HotelBookingAnalyzer(file_path='hotel_bookings.csv')
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.perform_eda()
    X_processed, y_target = analyzer.feature_engineer()  # Ensure this runs and populates attributes

    if X_processed is not None:  # Or check hasattr(analyzer, 'X_processed')
        print("\nSample of processed features (X) head:")
        print(analyzer.X_processed.head())  # Use the class attribute
        print("\nSample of target variable (y) head:")
        print(analyzer.y_target.head())  # Use the class attribute

        analyzer.build_and_evaluate_model()
    else:
        print("Feature engineering did not produce data. Model building skipped.")