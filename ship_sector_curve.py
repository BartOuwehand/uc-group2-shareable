import numpy as np

def plume_triangle_mask(data, uncertainty_line, ship_pixel_first, ship_pixel_last):
    ''' Apply a mask to data that selects the pixels that are inside the plume triangle '''
    mask = np.zeros_like(data)
    for unc_idx, img_idx in enumerate(np.arange(ship_pixel_last[1], ship_pixel_first[1])): # along the plume
        # check if the uncertainty line is within the cropped image; if not, set the bounds to the image bounds
        if ship_pixel_first[0]-uncertainty_line[unc_idx] < 0:
            lower_bound = 0
        elif ship_pixel_first[0]+uncertainty_line[unc_idx] > mask.shape[0]:
            upper_bound = mask.shape[0]
        else:
            lower_bound = ship_pixel_first[0]-uncertainty_line[unc_idx]
            upper_bound = ship_pixel_first[0]+uncertainty_line[unc_idx]+1
        mask[lower_bound:upper_bound, img_idx] = 1

    masked_data = np.zeros_like(data)*np.nan
    masked_data[mask==1] = data[mask==1]
    return masked_data

def get_uncertainty_line(x, wind_uncertainty=20):
    ''' Get the uncertainty line for the plume triangle; pixel indices for a line coming from the origin with a certain angle '''
    y = np.tan(np.deg2rad(wind_uncertainty)) * x
    # round to nearest integer
    y = np.round(y).astype(int)
    return y

def get_plume_curve(no2_rotated_clipped, no2e_rotated_clipped, no2_rotated_sectored, no2e_rotated_sectored, uncertainty_line, buffer_length):
    ''' Get the plume curve; it consists of: 
            - the plume in the ship sector, found with max of the column density in each column in the ship sector
            - the background on the left and right of the ship sector, found with the mean of the column density in columns outside the ship sector 
    '''	

    signal = np.zeros(no2_rotated_clipped.shape[1])
    signal_err = np.zeros(no2_rotated_clipped.shape[1])
    no2_sector = no2_rotated_sectored[:,buffer_length[0]:-buffer_length[0]]
    no2e_sector = no2e_rotated_sectored[:,buffer_length[0]:-buffer_length[0]]
    try:
        plume_maxidx = np.nanargmax(no2_sector, axis=0)
        signal[:buffer_length[0]] = np.nanmean(no2_rotated_clipped[:,:buffer_length[0]], axis=0) # background on the left
        signal[-buffer_length[0]:] = np.nanmean(no2_rotated_clipped[:,-buffer_length[0]:], axis=0) # background on the right
        signal[buffer_length[0]:-buffer_length[0]] = no2_sector[plume_maxidx,np.arange(no2_sector.shape[1])] # plume signal
        signal_err[:buffer_length[0]] = np.nanmean(no2e_rotated_clipped[:,:buffer_length[0]], axis=0)
        signal_err[-buffer_length[0]:] = np.nanmean(no2e_rotated_clipped[:,-buffer_length[0]:], axis=0)
        signal_err[buffer_length[0]:-buffer_length[0]] = no2e_sector[plume_maxidx,np.arange(no2e_sector.shape[1])]

        if np.sum(~np.isnan(signal)) != np.sum(~np.isnan(signal_err)):
            print('signal and signal_err have different number of non-nan values')
            return None, None, None
        
    except ValueError: # if all values are nan in a column - ideally we fix this so it just returns nan values for those columns
        print('all values are nan in a column')
        return None, None, None
        
    indices = np.arange(no2_rotated_clipped.shape[1], dtype=int)
    return signal, signal_err, indices