import numpy as np



def logs_to_pdfs(logs, bins, nbins=None, range_limits=None, bin_spacing='log'):
    """Computes the PDF of errors for each sample using an ensemble of error trajectories.
    Modified from SO post:
    
    https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
    
    Args:        
        logs (list[np.ndarray]): a list of M arrays, where each element in the
            list is of size (T_m, N). T_m is the number of training epochs for
            the m-th model in the ensemble.
            
        bins (np.ndarray): array of bin edges. If not provided, must provide
        `nbins`, `range_limits`, and `bin_spacing`.

        nbins (int): the number of bins to use for computing the PDFs
        
        range_limits (tuple[float]): the upper/lower limits of the bins. If not
        provided, uses the min/max error from `logs`.
        
        bin_spacing (str): one of "log" or "linear". Default is "log".

    Returns:
        counts, bins
    """

    if not isinstance(logs, list):
        logs = [logs]
    
    data2D = np.concatenate(logs, axis=0).T  # expected shape is (N, total_num_epochs)
    
    if range_limits == None:
        range_limits = (data2D.min(), data2D.max())
        
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    
    if bins is None:
        if bin_spacing == 'log':
            bins = np.logspace(np.log10(R[0]), np.log10(R[1]), nbins+1)
        elif bin_spacing == 'linear':
            bins = np.linspace(R[0], R[1], nbins+1)
        else:
            raise NotImplementedError("Only 'log' and 'linear' are supported values for `bin_spacing`")
    
    idx = np.searchsorted(bins, data2D,'right')-1

    nbins = bins.shape[0]-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==nbins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = nbins*np.arange(data2D.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = nbins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data2D.shape[:-1] + (nbins,)

    # Normalize
    counts = counts.astype(data2D.dtype)
    bin_widths = np.diff(bins)
    counts /= bin_widths[None, :]
    total = np.sum(counts*bin_widths[None, :], axis=1)
    mask = total > 0.0
    counts[mask] /= total[mask, None]  # to avoid divide by zero

    return counts, bins


def logs_to_uncertainties(logs, atol, rtol=0.0, magnitudes=None, normalization=None, collapse=True):
    """Compute pointwise uncertainties using an ensemble of error trajectories.
    
    Args:
        logs (list[np.ndarray]): a list of M arrays, where each element in the
            list is of size (T_m, N). T_m is the number of training epochs for
            the m-th model in the ensemble.

        atol (float or np.ndarray): the absolute tolerance threshold for
            calculating uncertainty. If a float is given, the same threshold is
            used for all samples. If an array is given it should be a vector of
            shape (N,) corresponding to the tolerances on each sample.

        rtol (float or np.ndarray): the relative tolerance threshold for
        calculating uncertainty. Default is 0.

        magnitudes (np.ndarray): a length-N array of the magnitudes of the
            target values. Cannot be None if rtol>0.

        normalization (np.ndarray): a vector of shape (N,). Normalization values
            to apply before applying the tolerance thresholds.

        collapse (bool): if False, returns the uncertainties for each M array of
            error trajectories. Useful for 'cartography' plots.

    """
    if normalization is None:
        normalization = np.ones(logs[0].shape[1])

    if isinstance(atol, (int, float)):
        atol = np.ones(logs[0].shape[1])*atol

    if isinstance(rtol, (int, float)):
        rtol = np.ones(logs[0].shape[1])*rtol

    if len(logs[0].shape) == 3:
        # add dummy dimension for xyz
        normalization = normalization.reshape(-1, 1)
        atol = atol.reshape(-1, 1)
        rtol = rtol.reshape(-1, 1)

    uncertainties = []
    for err in logs:
        if magnitudes is not None:
            correct = (err/normalization < (atol + rtol*np.abs(magnitudes)))
        else:
            if rtol.any() > 0: raise RuntimeError('Cannot specify rtol if true is None')
            correct = err/normalization < atol

        uncertainties.append(np.average(correct, axis=0))

    if collapse:
        return 1-np.average(np.stack(uncertainties), axis=0)
    else:
        return np.stack([1-u for u in uncertainties])


def bin_uq_by_group(df, group_key, error_key='error', uq_key='uq'):
    """
    Convenience function for plotting calibration curves.

    ```
    bins, binned_errors = bin_uq_by_group(df, group_key='label', error_key='force_error', uq_key='force_uq')

    for g, binned in binned_errors.items():
        plt.plot(bins[1:], [np.average(_>atol*s) if len(_) else np.nan for _ in binned], label=g)
    ```
    """

    bins = np.linspace(0, 1, 101)

    binned_errors = {}
    for group, group_df in df.groupby(group_key):
        b = np.digitize(group_df[uq_key].values, bins)
        binned_errors[group] = [group_df[error_key].values[b-1==i] for i in range(bins.shape[0]-1)]

    b = np.digitize(df[uq_key].values, bins)
    binned_errors['Total'] = [df[error_key].values[b-1==i] for i in range(bins.shape[0]-1)]

    return bins, binned_errors
