def tweedie(mu,p,phi):
    """ Draw sample from a Tweedie distribution
        We restrict p to be between 1 and 2 (exclusive)
        so this is specifically a compound Poisson-Gamma distribution
        (p=1 corresponds to Poisson, p=2 corresponds to Gamma)

    Parameters
    ----------
    mu : float
        The mean of the distribution
    p : float
        The power parameter of the distribution
    phi : float
        The dispersion parameter of the distribution
    Returns
    -------
    x : Sampled value from specified Tweedie distribution. Non-negative float

    Examples
    --------
    
    >>> from numpy.random import seed
    >>> seed(12345)
    >>> [tweedie(10,1.3,50) for _ in range(10)]
    [40.454075903450345,
     0,
     39.282601493739854,
     0,
     0,
     47.32187189667185,
     0,
     37.11092528612381,
     0,
     0]
    
    """
    
    from numpy.random import poisson, gamma

    if(p<=1 or p>=2):
        print('p must be between (1,2)')
        return None
    
    # mean of poisson distribution
    lambd = mu**(2-p) / (phi*(2-p))
    
    # shape parameter of gamma distribution
    kappa = (2-p) / (p-1)
    
    # scale parameter of gamma distribution
    theta = phi * (p-1) * mu**(p-1)
    
    # Sample from Poisson distribution
    N = poisson(lambd)
    
    # Sample from gamma distribution for each Poisson event
    x = sum([gamma(kappa,theta) for _ in range(N)])
    
    return x
