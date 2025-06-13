from plavchan import __cuda__plavchan_pgram, get_device_count # type: ignore
import threading

def cuda_wrapper(mags, times, trial_periods, width, device, returnlist):
    returnlist[device] = __cuda__plavchan_pgram(mags, times, trial_periods, width, device)

def plavchan_periodogram(mags, times, trial_periods, width, device=0):
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

    # ensure that all inputs are lists
    if not isinstance(mags, list) or not all(isinstance(m, list) for m in mags):
        raise ValueError("mags must be a list of lists")
    if not isinstance(times, list) or not all(isinstance(t, list) for t in times):
        raise ValueError("times must be a list of lists")
    if not isinstance(trial_periods, list):
        raise ValueError("trial_periods must be a list")
    if not isinstance(width, (int, float)):
        raise ValueError("width must be a float or int")
    if not isinstance(device, int):
        raise ValueError("device must be an int")
    
    # ensure that all inner lists have the same length
    if len(mags) != len(times):
        raise ValueError("Mismatch in number of times and mags provided")
    for i in range(len(mags)):
        if len(mags[i]) != len(times[i]):
            raise ValueError(f"Mismatch in number of times and mags for object {i}")

    if len(trial_periods) == 0:
        raise ValueError("trial_periods must not be empty")
    if 0 in trial_periods:
        raise ValueError("trial_periods must not include 0")
    
    if width <= 0 or width > 1:
        raise ValueError("width must be on (0,1]")
    

    # zero out times

    for i in range(len(times)):
        min_t = min(times[i])
        times[i] = [t - min_t for t in times[i]]

    if device != -1: # Simple case
        retval = __cuda__plavchan_pgram(mags, times, trial_periods, width, device)
        if not retval:
            raise ValueError("C/CUDA Extension error.")
        return retval
    
    # Multi-threaded case
    n_devices = get_device_count()
    jobs = []
    results = [None] * n_devices
    print("Using", n_devices, "devices")
    for device_id in range(min(n_devices, len(mags))): # Use the minimum of n_devices and N to avoid overloading
        start = (device_id*len(mags)) // n_devices
        end = ((device_id+1)*len(mags)) // n_devices
        print("Device", device_id, "processing objects", start, "to", end)
        
        job = threading.Thread(
            target=cuda_wrapper,
            args=(mags[start:end], times[start:end], trial_periods, width, device_id, results),
        )
        jobs.append(job)
        job.start()
    
    jobs = [job.join() for job in jobs] # now a list of periodograms for each slice of total objects
    pgram = []
    for device_id, partial_pgram in enumerate(results):
        if partial_pgram is None:
            raise ValueError("C/CUDA Extension error: Error between objects {} and {}.".format(
                (device_id*len(mags)) // n_devices, ((device_id+1)*len(mags)) // n_devices))
        
        pgram.extend(partial_pgram)

    return pgram
