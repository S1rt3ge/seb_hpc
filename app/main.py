"""
Complete FastAPI App with Real Predictions
FIXED: Uses correct feature exclusion for 26-feature models
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import sqlite3
import pandas as pd
import sys
import os

# Add prediction module to path
sys.path.append('../prediction')

from predict_cashflow import CashFlowPredictor
from complete_feature_engine import CompleteFeatureEngine

app = FastAPI(title="SEB SME Cash Management")

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/SEB Login_files", StaticFiles(directory="../templates/SEB Login_files"), name="seb_login_files")

# Templates
templates = Jinja2Templates(directory="../templates")

# Global variables for models
predictor = None
feature_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global predictor, feature_engine

    print("=" * 80)
    print("INITIALIZING SEB CASH MANAGEMENT APP")
    print("=" * 80)

    try:
        predictor = CashFlowPredictor(models_dir='../prediction/models_dl_optimized')
        print("✓ Predictor loaded")

        feature_engine = CompleteFeatureEngine(scaler_params_path='scaler_params.json')
        print("✓ Feature engine loaded")
        print("\n⚠️  NOTE: Works with NEW customers - no pre-computed features needed!")

        print("=" * 80)
        print("✓ APP READY!")
        print("=" * 80)

    except Exception as e:
        print(f"ERROR during startup: {e}")
        raise


def get_db_connection():
    """Get SQLite database connection"""
    db_path = 'C:/Users/aruti/PycharmProjects/susliki-seb-hpc/db/transactions.db'
    if not os.path.exists(db_path):
        db_path = '../db/transactions.db'

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_customers_list():
    """Get list of unique customers from database"""
    conn = get_db_connection()

    query = """
        SELECT cust_id,
               COUNT(*) as txn_count,
               MIN(BookingDatetime) as first_txn,
               MAX(BookingDatetime) as last_txn
        FROM transactions
        GROUP BY cust_id
        HAVING txn_count >= 20
        ORDER BY txn_count DESC
        LIMIT 500
    """

    customers = pd.read_sql_query(query, conn)
    conn.close()

    customers_list = []
    for _, row in customers.iterrows():
        customers_list.append({
            'id': row['cust_id'],
            'name': f"Customer {row['cust_id'][:8]}... ({row['txn_count']} txns)",
            'txn_count': row['txn_count']
        })

    return customers_list


def get_customer_transactions(cust_id):
    """Get all transactions for a customer"""
    conn = get_db_connection()

    query = """
        SELECT *
        FROM transactions
        WHERE cust_id = ?
        ORDER BY BookingDatetime
    """

    transactions = pd.read_sql_query(query, conn, params=(cust_id,))
    conn.close()

    return transactions


def get_customer_predictions(cust_id, n_weeks=4):
    """
    Get predictions for a customer
    FIXED: Excludes cash_flow_growth from features (26 features, not 27)
    """

    print(f"\n{'=' * 80}")
    print(f"GENERATING PREDICTION FOR: {cust_id}")
    print(f"{'=' * 80}")

    # Step 1: Get transactions
    print("Step 1: Fetching transactions from database...")
    transactions = get_customer_transactions(cust_id)

    if len(transactions) < 20:
        raise ValueError(f"Customer {cust_id} has insufficient transaction history ({len(transactions)} txns)")

    print(f"  ✓ Found {len(transactions)} transactions")

    # Step 2: Calculate features
    print("Step 2: Engineering features from raw data...")
    feature_result = feature_engine.prepare_features_for_prediction(cust_id, transactions)

    features_df = feature_result['features_df']
    cluster = feature_result['cluster']
    last_cashflow = feature_result['last_cashflow']

    print(f"  ✓ Cluster: {cluster}")
    print(f"  ✓ Features: {features_df.shape}")

    # Step 3: Prepare feature matrix - CRITICAL FIX
    print("Step 3: Preparing features for model...")

    # FIXED: Exclude both cash_flow AND cash_flow_growth (target variable)
    exclude_cols = ['cust_id', 'cluster', 'cash_flow', 'cash_flow_growth', 'datetime']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[feature_cols].fillna(0).values

    print(f"  ✓ Feature matrix: {X.shape}")  # Should be (N, 26) not (N, 27)
    print(f"  ✓ Feature columns: {len(feature_cols)}")

    # Step 4: Get predictions
    print(f"Step 4: Generating predictions with model...")
    predictions_std, model_used = predictor.predict_best(cluster, X)

    print(f"  ✓ Model used: {model_used}")

    # Step 5: Convert to EUR
    print("Step 5: Converting to EUR...")
    growth_eur = predictor.to_euros(predictions_std[-n_weeks:])

    # Step 6: Generate future cash flows
    print("Step 6: Calculating future cash flows...")
    future_cashflows = [last_cashflow]
    for growth in growth_eur:
        future_cashflows.append(future_cashflows[-1] + growth)

    # Step 7: Calculate metrics
    print("Step 7: Calculating metrics...")
    avg_weekly_flow = growth_eur.mean()
    positive_weeks = sum(1 for cf in future_cashflows[1:] if cf > 0)
    negative_weeks = n_weeks - positive_weeks

    # Investment opportunity
    min_buffer = 5000
    surplus = [max(0, cf - min_buffer) for cf in future_cashflows[1:]]
    total_investable = sum(surplus)

    # Potential interest (2% annual)
    annual_rate = 0.02
    weekly_rate = annual_rate / 52
    potential_interest = total_investable * weekly_rate * n_weeks

    print(f"  ✓ Predictions complete!")
    print(f"{'=' * 80}\n")

    return {
        'customer_id': cust_id,
        'cluster': cluster,
        'model_used': model_used,
        'last_cashflow': last_cashflow,
        'predictions': future_cashflows[1:],
        'growth_rates': growth_eur.tolist(),
        'avg_weekly_flow': avg_weekly_flow,
        'positive_weeks': positive_weeks,
        'negative_weeks': negative_weeks,
        'total_investable': total_investable,
        'potential_interest': potential_interest,
        'forecast_period': n_weeks,
        'static_features': feature_result['static_features']
    }


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page with customer selection"""
    customers = get_customers_list()
    return templates.TemplateResponse(
        "SEB Login.html",
        {"request": request, "customers": customers}
    )


@app.post("/login")
async def login(customer_id: str = Form(...)):
    """Handle login and redirect to dashboard"""
    return RedirectResponse(
        url=f"/dashboard?customer_id={customer_id}",
        status_code=303
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, customer_id: str):
    """Dashboard with REAL predictions"""

    try:
        # Get predictions
        predictions = get_customer_predictions(customer_id, n_weeks=4)

        # Get transaction summary
        transactions = get_customer_transactions(customer_id)
        total_transactions = len(transactions)
        date_range = f"{transactions['BookingDatetime'].min()} to {transactions['BookingDatetime'].max()}"

        # Prepare chart data - WEEKLY FORECAST
        forecast_labels = [f"Week {i + 1}" for i in range(predictions['forecast_period'])]
        forecast_values = [round(cf, 2) for cf in predictions['predictions']]

        # Current balance
        current_balance = round(predictions['last_cashflow'], 2)

        # Calculate recommendation
        if predictions['total_investable'] > 1000:
            recommendation = f"Transfer €{predictions['total_investable']:,.2f} to Savings Account"
            rec_type = "invest"
        elif current_balance < 0:
            recommendation = "Consider securing additional financing"
            rec_type = "warning"
        else:
            recommendation = "Maintain current cash position"
            rec_type = "maintain"

        # Render dashboard with REAL DATA
        return templates.TemplateResponse(
            "SME Cash Management - SEB.html",
            {
                "request": request,
                "customer_id": customer_id,
                "cluster": predictions['cluster'],
                "model_used": predictions['model_used'],
                "current_balance": current_balance,
                "avg_weekly_flow": round(predictions['avg_weekly_flow'], 2),
                "potential_interest": round(predictions['potential_interest'], 2),
                "total_investable": round(predictions['total_investable'], 2),
                "positive_weeks": predictions['positive_weeks'],
                "negative_weeks": predictions['negative_weeks'],
                "recommendation": recommendation,
                "rec_type": rec_type,
                "forecast_labels": forecast_labels,
                "forecast_values": forecast_values,
                "total_transactions": total_transactions,
                "date_range": date_range,
            }
        )

    except Exception as e:
        print(f"Error in dashboard: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/{customer_id}")
async def api_predict(customer_id: str, weeks: int = 4):
    """API endpoint for predictions"""
    try:
        predictions = get_customer_predictions(customer_id, n_weeks=weeks)
        return JSONResponse(content=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/customers")
async def api_customers():
    """API endpoint to get customer list"""
    customers = get_customers_list()
    return JSONResponse(content=customers)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)