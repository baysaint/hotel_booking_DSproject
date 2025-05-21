# Hotel Booking Cancellation Prediction

## Project Overview

This project analyzes a hotel booking dataset to understand the factors influencing booking cancellations and to build a predictive model that can identify bookings likely to be canceled. The goal is to practice Data Science principles and gain insights on booking cancellations and optimize booking strategies.

The project follows a standard data science workflow:
1.  **Data Loading and Cleaning:** Ingesting the dataset and handling missing values, outliers, and data type inconsistencies.
2.  **Exploratory Data Analysis (EDA):** Visualizing and summarizing the data to uncover patterns, trends, and relationships between various booking attributes and cancellation status.
3.  **Feature Engineering:** Creating new meaningful features from existing ones and transforming features into a format suitable for machine learning.
4.  **Model Building and Evaluation:** Training and evaluating several classification models (Logistic Regression and Random Forest) to predict booking cancellations, comparing their performance using various metrics.

## Dataset

The dataset used is the "Hotel Booking Demand" dataset, commonly found on platforms like Kaggle. It contains detailed information on bookings for a city hotel and a resort hotel, including booking dates, lead time, length of stay, guest demographics, room types, deposit types, and cancellation status.

*   **Source:** (e.g., Kaggle - Hotel Booking Demand Dataset - You can add a link if you have one)
*   **Original Shape:** 119,390 bookings, 32 columns
*   **Cleaned Shape:** 119,209 bookings, 32 columns (after removing invalid/erroneous entries)
*   **Target Variable:** `is_canceled` (1 if canceled, 0 if not)

## File Structure

```
.
├── hotel_bookings.csv        # Original dataset (or link to it)
├── cleaned_hotel_bookings.csv # Dataset after cleaning
├── main.py                 # Main Python script implementing the analysis
├── plots/                    # Directory containing saved EDA and model evaluation plots
│   ├── cancellation_status_by_hotel_type.png
│   ├── cancellation_rate_by_arrival_month.png
│   ├── distribution_of_lead_time_by_cancellation_status.png
│   ├── cancellation_rate_by_lead_time_bins.png
│   ├── distribution_of_adr_by_cancellation_status.png
│   ├── cancellation_rate_by_adr_bins.png
│   ├── cancellation_status_by_deposit_type.png
│   ├── cancellation_status_by_customer_type.png
│   ├── cancellation_status_by_market_segment.png
│   ├── cancellation_status_by_repeated_guest.png
│   ├── correlation_heatmap_of_relevant_numerical_features.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── top_20_feature_importances_(random_forest).png
│   └── roc_curves.png
├── requirements.txt          # Python dependencies
└── README.md                 # This file

```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/baysaint/hotel_booking_DSproject
    cd hotel_booking_DSproject
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Dataset:**
    Ensure the `hotel_bookings.csv` file is present in the root directory or update the `file_path` variable in `main.py`.

## How to Run

Execute the main Python script:
```bash
python main.py
```

The script will:
- Load and clean the data, saving cleaned_hotel_bookings.csv.

- Perform Exploratory Data Analysis, printing summaries to the console and saving plots to the plots/ directory.

- Perform Feature Engineering.

- Build, train, and evaluate Logistic Regression and Random Forest models, printing performance metrics and saving relevant plots.


### Key Findings from EDA

- **Overall Cancellation Rate**: Approximately 37.08% of bookings are canceled.
- **Hotel Type**: City hotels (~41.8%) have a significantly higher cancellation rate than Resort hotels (~27.8%).
- **Lead Time**: There's a strong positive correlation between lead time and cancellations. Bookings made further in advance are more likely to be canceled (e.g., bookings with 600+ days lead time have a ~98% cancellation rate).
- **Deposit Type**: Bookings with a "Non Refund" deposit have an extremely high cancellation rate (~99.4%), while "Refundable" and "No Deposit" bookings have much lower rates. This is a critical factor.
- **Total Special Requests**: Guests making more special requests are less likely to cancel (correlation of -0.23 with is_canceled).
- **Required Car Parking**: Guests requiring parking spaces are less likely to cancel.
- **Market Segment**: "Groups" segment bookings tend to have higher cancellation rates, while "Direct" and "Corporate" bookings have lower rates.
- **Arrival Month**: Cancellation rates show seasonality, being higher in late spring/early summer (e.g., June ~41.5%) and lower in winter months (e.g., January ~30.5%).
### Feature Engineering Highlights

- **New Features Created**: _total_stay_duration, total_guests, is_family, arrival_date_month_numeric_.
- **Categorical Encoding**: One-Hot Encoding was applied to features like _hotel, meal, market_segment, distribution_channel, deposit_type,_ and _customer_type_.
- **Numerical Scaling**: StandardScaler was applied to numerical features.
- **High Cardinality Features**: Features like _country, agent, reserved_room_type, and assigned_room_type_ were initially passed through the preprocessor and then dropped if they remained as object types (as a simplification for this iteration). More advanced encoding techniques could be applied to these in future work.

### Model Performance
Two models were trained and evaluated:

1. **Logistic Regression (Baseline)**:

   - Accuracy: ~80.96%
   - ROC AUC: ~0.853
   - F1-score (Canceled): 0.70
   - Key takeaway: Decent baseline, but struggles with recalling actual cancellations (Recall for Canceled class: 0.60).
   
2. **Random Forest Classifier**:

   - Accuracy: ~86.01%
   - ROC AUC: ~0.928
   - F1-score (Canceled): 0.80
   - Key takeaway: Significantly outperforms Logistic Regression, with better precision and notably higher recall for cancellations (Recall for Canceled class: 0.76). _class_weight='balanced'_ was used to help address class imbalance.
### Top Feature Importances (from Random Forest):
   1. lead_time
   2. adr (Average Daily Rate)
   3. deposit_type_Non Refund
   4. arrival_date_month_numeric
   5. total_of_special_requests

### Conclusion and Potential Business Applications
   The Random Forest model demonstrates strong performance in predicting hotel booking cancellations. Key drivers for cancellations include long lead times, the absence of a non-refundable deposit, and fewer special requests.
   Potential applications for a hotel include:

   - **Targeted Interventions**: Identifying high-risk bookings early (e.g., long lead time, no non-refundable deposit) and engaging with customers to confirm their stay or offer flexible alternatives.
   - **Dynamic Pricing/Deposit Policies**: Adjusting deposit policies based on predicted cancellation risk.
   - **Resource Management**: Better forecasting of actual occupancy based on predicted cancellations, leading to improved staffing and inventory management.

### Future Work and Potential Improvements
   - **Advanced Encoding for High Cardinality Features**: Implement techniques like Target Encoding or Top-N One-Hot Encoding for _country_ and _agent_.
   - **Hyperparameter Tuning**: Optimize Random Forest (and other models) using GridSearchCV or RandomizedSearchCV.
   - **Explore Other Models**: Experiment with Gradient Boosting models (XGBoost, LightGBM, CatBoost).
   - **Time Series Analysis**: If more granular daily/weekly booking data were available, time series forecasting could be used to predict demand and cancellations.
   - **Deployment**: Package the model into an API for integration with a hotel's booking system (using Flask/Django).
## Author
   Murat Gencoglu 
   
GitHub: baysaint

## License
   MIT License