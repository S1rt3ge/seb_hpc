"""
ENHANCED TRAINING OUTPUT
Shows loss curves, multiple metrics, and real EUR values
"""

import pandas as pd
import numpy as np
import json


def load_scaler_params():
    """Load scaler parameters for EUR conversion"""
    with open('scaler_params.json', 'r') as f:
        params = json.load(f)
    return params['target_mean'], params['target_std']


def descale_predictions(predictions_std, actuals_std, global_mean, global_std):
    """Convert standardized predictions/actuals back to EUR"""
    predictions_eur = predictions_std * global_std + global_mean
    actuals_eur = actuals_std * global_std + global_mean
    return predictions_eur, actuals_eur


def calculate_comprehensive_metrics(predictions_std, actuals_std, global_mean, global_std):
    """
    Calculate ALL useful metrics in both standardized and EUR units
    """
    # Convert to EUR
    predictions_eur, actuals_eur = descale_predictions(
        predictions_std, actuals_std, global_mean, global_std
    )

    # STANDARDIZED METRICS
    rmse_std = np.sqrt(np.mean((predictions_std - actuals_std) ** 2))
    mae_std = np.mean(np.abs(predictions_std - actuals_std))

    # EUR METRICS
    rmse_eur = np.sqrt(np.mean((predictions_eur - actuals_eur) ** 2))
    mae_eur = np.mean(np.abs(predictions_eur - actuals_eur))

    # DIRECTION ACCURACY (critical for cash flow!)
    pred_direction = np.sign(predictions_eur)
    actual_direction = np.sign(actuals_eur)
    direction_accuracy = np.mean(pred_direction == actual_direction)

    # PERCENTAGE ERRORS (where actual != 0)
    non_zero_mask = np.abs(actuals_eur) > 100  # Only for |actual| > €100
    if non_zero_mask.sum() > 0:
        pct_errors = np.abs((predictions_eur[non_zero_mask] - actuals_eur[non_zero_mask]) /
                            actuals_eur[non_zero_mask]) * 100
        mean_pct_error = np.mean(pct_errors)
        median_pct_error = np.median(pct_errors)
    else:
        mean_pct_error = None
        median_pct_error = None

    # R² SCORE
    ss_res = np.sum((actuals_std - predictions_std) ** 2)
    ss_tot = np.sum((actuals_std - np.mean(actuals_std)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # BIAS (systematic over/under prediction)
    bias_eur = np.mean(predictions_eur - actuals_eur)

    # PERCENTILE ERRORS (for robustness)
    errors_eur = np.abs(predictions_eur - actuals_eur)
    p50_error = np.median(errors_eur)
    p75_error = np.percentile(errors_eur, 75)
    p90_error = np.percentile(errors_eur, 90)
    p95_error = np.percentile(errors_eur, 95)

    return {
        # Standardized
        'rmse_std': rmse_std,
        'mae_std': mae_std,
        'r2': r2_score,

        # EUR (REAL MONEY)
        'rmse_eur': rmse_eur,
        'mae_eur': mae_eur,
        'bias_eur': bias_eur,

        # Direction
        'direction_accuracy': direction_accuracy,

        # Percentages
        'mean_pct_error': mean_pct_error,
        'median_pct_error': median_pct_error,

        # Error distribution
        'p50_error_eur': p50_error,
        'p75_error_eur': p75_error,
        'p90_error_eur': p90_error,
        'p95_error_eur': p95_error,
    }


def print_training_header(cluster_name, n_customers, n_observations):
    """Print nice header for training"""
    print("\n" + "=" * 80)
    print(f"TRAINING: {cluster_name}")
    print("=" * 80)
    print(f"Customers: {n_customers:,}")
    print(f"Observations: {n_observations:,}")
    print("=" * 80)


def print_epoch_progress(epoch, total_epochs, train_loss, val_loss, elapsed_time):
    """Print progress during training"""
    print(f"Epoch [{epoch:2d}/{total_epochs}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
          f"Time: {elapsed_time:.1f}s")


def print_model_results(model_name, metrics, is_best=False):
    """Print comprehensive results for a model"""
    marker = "★" if is_best else " "

    print(f"\n{marker} {model_name} Results:")
    print(f"  {'─' * 70}")

    # Standardized metrics
    print(f"  Standardized Metrics:")
    print(f"    RMSE: {metrics['rmse_std']:.6f}")
    print(f"    MAE:  {metrics['mae_std']:.6f}")
    print(f"    R²:   {metrics['r2']:.6f}")

    # EUR metrics (REAL MONEY!)
    print(f"\n  Real Money Metrics (EUR):")
    print(f"    RMSE: €{metrics['rmse_eur']:>12,.2f}  ← Average prediction error")
    print(f"    MAE:  €{metrics['mae_eur']:>12,.2f}  ← Median-like error")
    print(f"    Bias: €{metrics['bias_eur']:>12,.2f}  ← Systematic over/under prediction")

    # Direction accuracy
    print(f"\n  Direction Accuracy:")
    print(f"    {metrics['direction_accuracy'] * 100:>5.1f}%  ← Correct up/down prediction")

    # Percentage errors (if available)
    if metrics['mean_pct_error'] is not None:
        print(f"\n  Percentage Errors (for |actual| > €100):")
        print(f"    Mean:   {metrics['mean_pct_error']:>6.1f}%")
        print(f"    Median: {metrics['median_pct_error']:>6.1f}%")

    # Error distribution
    print(f"\n  Error Distribution (EUR):")
    print(f"    50% of errors < €{metrics['p50_error_eur']:>10,.2f}")
    print(f"    75% of errors < €{metrics['p75_error_eur']:>10,.2f}")
    print(f"    90% of errors < €{metrics['p90_error_eur']:>10,.2f}")
    print(f"    95% of errors < €{metrics['p95_error_eur']:>10,.2f}")


def print_sample_predictions(predictions_eur, actuals_eur, n_samples=10):
    """Show sample predictions vs actuals"""
    print(f"\n  Sample Predictions (first {n_samples}):")
    print(f"    {'Actual (EUR)':>15} {'Predicted (EUR)':>18} {'Error (EUR)':>15} {'Error %':>10}")
    print(f"    {'─' * 15} {'─' * 18} {'─' * 15} {'─' * 10}")

    for i in range(min(n_samples, len(predictions_eur))):
        actual = actuals_eur[i]
        pred = predictions_eur[i]
        error = pred - actual

        if abs(actual) > 10:
            error_pct = abs(error / actual) * 100
        else:
            error_pct = 0

        print(f"    {actual:>15,.2f} {pred:>18,.2f} {error:>15,.2f} {error_pct:>9.1f}%")


def print_final_summary(all_results, best_model):
    """Print final comparison of all models"""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - MODEL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<15} {'RMSE (EUR)':>15} {'MAE (EUR)':>15} {'Direction Acc':>15} {'R²':>10}")
    print("─" * 75)

    for model_name in ['KNN', 'LSTM', 'GRU', 'Transformer']:
        if model_name in all_results:
            metrics = all_results[model_name]
            marker = "★" if model_name == best_model else " "

            print(f"{marker} {model_name:<13} "
                  f"€{metrics['rmse_eur']:>13,.2f} "
                  f"€{metrics['mae_eur']:>13,.2f} "
                  f"{metrics['direction_accuracy'] * 100:>13.1f}% "
                  f"{metrics['r2']:>9.3f}")

    print("\n★ = Best model (lowest RMSE in EUR)")


def save_detailed_results(cluster_name, all_results, predictions_dict, actuals_dict):
    """Save detailed results to CSV for analysis"""
    import os

    os.makedirs('training_results', exist_ok=True)

    # Save metrics summary
    summary_df = pd.DataFrame(all_results).T
    summary_df.to_csv(f'training_results/{cluster_name}_metrics_summary.csv')

    # Save predictions for best model
    best_model = min(all_results.keys(), key=lambda k: all_results[k]['rmse_eur'])

    pred_df = pd.DataFrame({
        'actual_eur': actuals_dict[best_model],
        'predicted_eur': predictions_dict[best_model],
        'error_eur': predictions_dict[best_model] - actuals_dict[best_model],
        'abs_error_eur': np.abs(predictions_dict[best_model] - actuals_dict[best_model]),
        'correct_direction': np.sign(predictions_dict[best_model]) == np.sign(actuals_dict[best_model])
    })

    pred_df.to_csv(f'training_results/{cluster_name}_{best_model}_predictions.csv', index=False)

    print(f"\n✓ Results saved to training_results/{cluster_name}_*")


# Example usage template for integration into training script:
USAGE_EXAMPLE = """
# In your training loop:

global_mean, global_std = load_scaler_params()

print_training_header(cluster_name, n_customers, n_observations)

# During training epochs:
for epoch in range(MAX_EPOCHS):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    print_epoch_progress(epoch, MAX_EPOCHS, train_loss, val_loss, elapsed_time)

# After training each model:
predictions_std, actuals_std = evaluate_model(model, test_loader)
metrics = calculate_comprehensive_metrics(predictions_std, actuals_std, global_mean, global_std)
print_model_results(model_name, metrics, is_best=False)

predictions_eur, actuals_eur = descale_predictions(predictions_std, actuals_std, global_mean, global_std)
print_sample_predictions(predictions_eur, actuals_eur)

# After all models:
print_final_summary(all_results, best_model)
save_detailed_results(cluster_name, all_results, predictions_dict, actuals_dict)
"""

print("Enhanced training output module loaded!")
print("Functions available:")
print("  - load_scaler_params()")
print("  - calculate_comprehensive_metrics()")
print("  - print_training_header()")
print("  - print_epoch_progress()")
print("  - print_model_results()")
print("  - print_sample_predictions()")
print("  - print_final_summary()")
print("  - save_detailed_results()")