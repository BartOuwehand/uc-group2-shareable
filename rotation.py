import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def rotateImage(img, angle, pivot):
    ''' Rotate image about a pivot point (pads the image to rotate about the center)
    '''
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    # imgR = scipy.ndimage.rotate(imgP, angle, reshape=False, order=3, cval=np.nan, prefilter=True)
    imgR = scipy.ndimage.rotate(imgP, angle, reshape=False, order=1, cval=np.nan, prefilter=True)
    # imgR = scipy.ndimage.rotate(imgP, angle, reshape=False, order=0, cval=np.nan, prefilter=True)
    return imgR

def perform_rotation(no2, no2e, angle, pivot):
    ''' Rotate the image so that the track is aligned with the x and y axis for each vessel in the box
    '''
    
    no2[np.isnan(no2)] = 0 # replace nan values with zero so we can do spline interpolation when rotating (instead of only nearest neighbour)
    no2e[np.isnan(no2)] = 0 # replace nan values with zero so we can do spline interpolation when rotating (instead of only nearest neighbour)
    
    # rotate the image so that the track is aligned with the x and y axis for each vessel in the box
    no2_rotated = rotateImage(no2, -angle, pivot)
    no2e_rotated = rotateImage(no2e, -angle, pivot)

    no2_rotated[no2_rotated == 0] = np.nan # remove zero values, which were NaN before rotation
    no2e_rotated[no2e_rotated == 0] = np.nan # remove zero values, which were NaN before rotation

    return no2_rotated, no2e_rotated

def get_pivot_pixel(AIS_handler, mmsi, no2_sliced_xds):
    ''' Get the pivot pixel for the rotation - use the pixel closest to the ship's first position 
    '''

    ship_lon_first = AIS_handler.AIS_data[mmsi]['longitude_shifted'][0] ; ship_lat_first = AIS_handler.AIS_data[mmsi]['Latitude_shifted'][0]
    ship_lon_last = AIS_handler.AIS_data[mmsi]['longitude_shifted'][-1] ; ship_lat_last = AIS_handler.AIS_data[mmsi]['Latitude_shifted'][-1]
    ship_pixel_first = [np.argmin(np.abs(ship_lon_first - no2_sliced_xds.longitude.data)), np.argmin(np.abs(ship_lat_first - no2_sliced_xds.latitude.data))]
    ship_pixel_last = [np.argmin(np.abs(ship_lon_last - no2_sliced_xds.longitude.data)), np.argmin(np.abs(ship_lat_last - no2_sliced_xds.latitude.data))]

    return ship_pixel_first, ship_pixel_last

def get_pixelsize(pixelsize_lat, pixelsize_lon, angle, no2_sliced_xds):
    ''' Get the pixel size in the rotated image 
    '''

    no2_sliced_img = no2_sliced_xds.data[0,:,:]
    rotated_fullimg = scipy.ndimage.rotate(no2_sliced_img, angle, reshape=True, order=1, cval=np.nan, prefilter=True)

    original_height = no2_sliced_img.shape[0] * pixelsize_lat # height of the original image in km
    original_width = no2_sliced_img.shape[1] * pixelsize_lon # width of the original image in km

    rotated_width = np.abs( np.cos(np.deg2rad(angle))*original_width ) + np.abs( np.sin(np.deg2rad(angle))*original_height )
    rotated_height = np.abs( np.sin(np.deg2rad(angle))*original_width ) + np.abs( np.cos(np.deg2rad(angle))*original_height ) # width and height of the rotated image in km

    rotated_height_pixels, rotated_width_pixels = rotated_fullimg.shape # width and height of the rotated image in pixels

    pixel_size_x = rotated_width / rotated_width_pixels
    pixel_size_y = rotated_height / rotated_height_pixels    

    return pixel_size_x, pixel_size_y