from plavchan_gpu import __cuda__plavchan_pgram

def plavchan_periodogram(mags, times, trial_periods, width):
    """
    Calculate the Plavchan periodogram using CUDA GPU acceleration.
    
    Parameters
    ----------
    mags : list of lists of floats
        Magnitude measurements for each object
    times : list of lists of floats
        Time measurements for each object
    trial_periods : list of floats
        Trial periods to calculate the periodogram for. Should not include 0.
    width : float
        The phase width parameter
        
    Returns
    -------
    list of lists of floats
        Periodogram values for each object and trial period
    """
    return __cuda__plavchan_pgram(mags, times, trial_periods, width)

__version__ = "0.1.0"