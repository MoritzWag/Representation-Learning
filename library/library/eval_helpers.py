import torch
import math  

def log_density_gaussian(x, mu, logvar):
    """Calculates the log density of a gaussian.

    Args:
        x: {torch.Tensor} value at which to calculate the log density
        mu: {torch.Tensor} mean
        logvar: {torch.Tensor, np.array, float} log variance
    
    Returns:
        log_density {torch.Tensor}:
    """
    normalization = 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)

    return log_density


