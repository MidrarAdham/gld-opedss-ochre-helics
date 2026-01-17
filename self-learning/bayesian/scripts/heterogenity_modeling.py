# %%
# pypi packages
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize
# my packages
import bayesian_plots as my_baysian_vis
# %%
def extract_final_thetas(all_histories_seq):
    """
    Extract final theta estimate values from sequential bayesian results
    
    :param all_histories_seq: Dict of {wh_name: history_dict}
    :return: Dict with WH names and their final theta values
    """
    final_thetas = {}
    
    for wh_name, history in all_histories_seq.items():
        final_theta = history['mean'][-1]  # Last mean value
        final_alpha = history['alpha'][-1]
        final_beta = history['beta'][-1]
        
        final_thetas[wh_name] = {
            'theta': final_theta,
            'alpha': final_alpha,
            'beta': final_beta
        }
    
    return final_thetas

def fit_beta_distribution(theta_values):
    """
    Fit a Beta distribution to the observed theta values from the previous methods.
    The best way to track the process of these implementations si to follow the master methods
    
    :param theta_values: List or array of theta values
    :return: Fitted parameters (alpha, beta) and fitted distribution
    """
    theta_array = np.array(theta_values)
    
    # first method: Method of Moments
    mean_theta = np.mean(theta_array)
    var_theta = np.var(theta_array)
    
    # Beta distribution moments:
    # mean = alpha / (alpha + beta)
    # var = alpha x beta / ((alpha + beta)^2 (alpha + beta + 1))
    
    # Solve for alpha and beta
    if var_theta >= mean_theta * (1 - mean_theta):
        print("Warning: Variance too high for Beta distribution")
        var_theta = mean_theta * (1 - mean_theta) * 0.99
    
    common = mean_theta * (1 - mean_theta) / var_theta - 1
    alpha_mom = mean_theta * common
    beta_mom = (1 - mean_theta) * common
    
    # second method: Maximum Likelihood Estimation (said to be more accurate) - need to study both methods
    def neg_log_likelihood(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        return -np.sum(beta.logpdf(theta_array, a, b))
    
    # Use method of moments as initial guess
    result = minimize(neg_log_likelihood, 
                     x0=[alpha_mom, beta_mom],
                     method='L-BFGS-B',
                     bounds=[(0.1, 1000), (0.1, 1000)])
    
    alpha_mle, beta_mle = result.x

    return {
        'alpha_mom': alpha_mom,
        'beta_mom': beta_mom,
        'alpha_mle': alpha_mle,
        'beta_mle': beta_mle,
        'mean_empirical': mean_theta,
        'std_empirical': np.std(theta_array)
    }


def predict_aggregate_with_heterogeneity(n_units, fitted_params, kw_per_unit=4.5):
    """
    Predict aggregate load accounting for heterogeneity
    
    :param n_units: Number of water heaters to aggregate
    :param fitted_params: Fitted Beta distribution parameters
    :param kw_per_unit: kW consumption when ON
    :return: Predicted load statistics
    """
    alpha_mle = fitted_params['alpha_mle']
    beta_mle = fitted_params['beta_mle']
    
    # Sample theta values from fitted distribution
    n_samples = 10000
    theta_samples = beta.rvs(alpha_mle, beta_mle, size=(n_samples, n_units))
    
    # For each sample, calculate expected number ON
    expected_on = theta_samples.sum(axis=1)  # Sum across units
    expected_kw = expected_on * kw_per_unit
    
    # Statistics
    mean_kw = np.mean(expected_kw)
    std_kw = np.std(expected_kw)
    ci_lower = np.percentile(expected_kw, 2.5)
    ci_upper = np.percentile(expected_kw, 97.5)
    
    return {
        'mean_kw': mean_kw,
        'std_kw': std_kw,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'samples': expected_kw
    }

def heterogeneity_modeling(all_histories_seq):
    """
    Model heterogeneity in theta values across water heaters
    """
    
    # Extract final thetas
    final_theta_data = extract_final_thetas(all_histories_seq)
    theta_values = [data['theta'] for data in final_theta_data.values()]
    
    print(f"\nNumber of water heaters: {len(theta_values)}")
    print(f"Mean {r'$\theta$'}: {np.mean(theta_values):.4f}")
    print(f"Std {r'$\theta$'}:  {np.std(theta_values):.4f}")
    print(f"Min {r'$\theta$'}:  {np.min(theta_values):.4f}")
    print(f"Max {r'$\theta$'}:  {np.max(theta_values):.4f}")
    
    # Fit Beta distribution
    fitted_params = fit_beta_distribution(theta_values)
    
    # Visualizations
    my_baysian_vis.plot_heterogeneity_distribution(theta_values, fitted_params)
    my_baysian_vis.plot_qq_plot(theta_values, fitted_params)
    my_baysian_vis.analyze_percentiles(theta_values, fitted_params)
    
    # Predictions
    for n_units in [100, 500, 1000, 5000]:
        predict_aggregate_with_heterogeneity(n_units, fitted_params)
    
    return fitted_params