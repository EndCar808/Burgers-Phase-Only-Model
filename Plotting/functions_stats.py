import numpy as np



#######################
##   Function Defs   ##
#######################
def normalized_pdf(counts, bin_edges):

    ## Get indexes of non zero entries
    indx = np.argwhere(counts != 0)

    ## Calculate bin widths
    dx = bin_edges[1] - bin_edges[0]

    ## Get bin centres
    bin_centr = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    bin_centr = bin_centr[indx]

    ## Get the pdf
    pdf = counts[indx] / np.sum(counts * dx)

    ## Compute variance to normalize
    var = np.sqrt(np.sum(pdf * bin_centr ** 2 * dx))

    ## Normalize
    norm_pdf = pdf * var
    bin_centr /= var 
    dx /= var

    return norm_pdf, bin_centr, dx




def percentiles(counts, bins, percentile, pdf = False):

    ## Compute the normalized PDF if needed
    if pdf == True:
        pdf, cntrs, dx = normalized_pdf(counts, bins)
    else:
        pdf   = counts
        cntrs = bins
        dx    = bins[1] - bins[0]

    ## 
    left_tail  = np.where(cntrs <= -percentile, pdf, 0.0)
    right_tail = np.where(cntrs >= percentile, pdf, 0.0) 

    ## Calculate the percentiles by integrating
    percentiles_left  = np.sum(left_tail * dx)
    percentiles_right = np.sum(right_tail * dx)


    return percentiles_left, percentiles_right