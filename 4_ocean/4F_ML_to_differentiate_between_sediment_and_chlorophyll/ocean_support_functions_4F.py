# General packages;
# All functions authored by Ben Loveday (2021) unless otherwise specified, 
# please attribute appropriately.

import cartopy.crs as ccrs
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4 as nc
import numpy as np
import os
from scipy.ndimage.measurements import label
from skimage import exposure
import xarray as xr

# --- FUNCTIONS
def spheric_dist(lat1,lat2,lon1,lon2,mode="global"):       
    '''     
     Function dist=spheric_dist(lat1,lat2,lon1,lon2)
     compute distances for a simple spheric earth

     Inputs:
     lat1 : latitude of first point (array or point)
     lon1 : longitude of first point (array or point)
     lat2 : latitude of second point (array or point)
     lon2 : longitude of second point (array or point)

     Outputs:
     dist : distance from first point to second point (array)
      
     This function is adapted from the matlab implementation
     of the ROMS_AGRIF tools, distrubted under the GPU licence.
     (https://www.croco-ocean.org/download/roms_agrif-project/)
    '''
    R = 6367442.76
    # Determine proper longitudinal shift.
    l = np.abs(lon2-lon1)
    try:
        l[l >= 180] = 360 - l[l >= 180]
    except:
        pass
    # Convert Decimal degrees to radians.
    deg2rad = np.pi/180
    phi1    = (90-lat1)*deg2rad
    phi2    = (90-lat2)*deg2rad
    theta1  = lon1*deg2rad
    theta2  = lon2*deg2rad
 
    lat1    = lat1*deg2rad
    lat2    = lat2*deg2rad
    l       = l*deg2rad

    if mode=="global":
        # Compute the distances: new
        cos     = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) + 
                   np.cos(phi1)*np.cos(phi2))
        arc     = np.arccos( cos )
        dist    = R*arc
    elif mode=="regional":
        # Compute the distances: 1 old, deprecated ROMS version - unsuitable for global
        dist    = R*np.arcsin(np.sqrt(((np.sin(l)*np.cos(lat2))**2) + (((np.sin(lat2)*np.cos(lat1)) - \
                  (np.sin(lat1)*np.cos(lat2)*np.cos(l)))**2)))
    elif mode=="local":
        #uses approx for now: 
        x = [lon2-lon1] * np.cos(0.5*[lat2+lat1])
        y = lat2-lat1
        dist = R*[x*x+y*y]^0.5
    else:
        print("incorrect mode")

    return dist

def get_coords(LON, lonmin, lonmax, LAT,latmin, latmax, nearest_flag):
    '''
     Find I and J corners of points of a box in a lat/lon grid
     
     Inputs:
     LON/LAT :            longitude/latitude grid
     lonmin/lonmax :      minimum/maximum longitude
     latmin/latmax :      minimum/maximum latitude
     nearest_flag :       activate to select a point, not a box

     Outputs:
     I1f, I2f, J1f, J2f : box corner points
    '''
    dist_i1 = spheric_dist(latmin,LAT,lonmin,LON)
    #J is the X-coord
    I1,J1   = np.where(dist_i1 == np.nanmin(dist_i1))

    if nearest_flag:
        I1 = I1[0]
        J1 = J1[0]
        I2 = I1+1
        J2 = J1+1
    else:
        dist_i2 = spheric_dist(latmax,LAT,lonmax,LON)
        I2, J2   = np.where(dist_i2 == np.nanmin(dist_i2))
        I1 = I1[0]
        J1 = J1[0]
        I2 = I2[0]
        J2 = J2[0]

    # re-arrange coordinates so that we count upwards...
    if J2 < J1:
        J1f = J2
        J2f = J1
    else:
        J1f = J1
        J2f = J2

    if I2 < I1:
        I1f = I2
        I2f = I1
    else:
        I1f = I1
        I2f = I2
    
    return I1f, I2f, J1f, J2f

def subset_image(grid_lat, grid_lon, plot_extents):
    '''
     Cuts a box out of an image using the grid indices
     for the image corners. BEWARE USING THIS ON HALF-ORBIT,
     FULL-ORBIT or POLAR DATA. Uses spheric distance 
     calculator to find nearest points.
     
     Inputs:
     grid_lat :       the latitude grid
     grid_lon :       the longitude grid
     plot_extents :   the extent of the area you want [lon1,lon2,lat1,lat2]
     
     Outputs:
     
     i1, i2, j1, j1 : the i/j coordinates of the corners of the requested box
     
    '''
    # bottom left
    dist = spheric_dist(plot_extents[2], grid_lat, plot_extents[0], grid_lon)
    i0, j0 = np.unravel_index(dist.argmin(), dist.shape)
    
    # bottom right
    dist = spheric_dist(plot_extents[2], grid_lat, plot_extents[1], grid_lon)
    i1, j1 = np.unravel_index(dist.argmin(), dist.shape)    
    
    # top right
    dist = spheric_dist(plot_extents[3], grid_lat, plot_extents[1], grid_lon)
    i2, j2 = np.unravel_index(dist.argmin(), dist.shape)
    
    # top left
    dist = spheric_dist(plot_extents[3], grid_lat, plot_extents[0], grid_lon)
    i3, j3 = np.unravel_index(dist.argmin(), dist.shape)
    
    return min([i0, i1, i2, i3]), max([i0, i1, i2, i3]), min([j0, j1, j2, j3]), max([j0, j1, j2, j3])
    
def reduce_image(grid, grid_factor):
    '''
     Re-samples an image on a coarser grid
     
     Inputs:
     grid :        the grid to be resampled
     grid_factor : the resampling factor
     
     Outputs:
     grid :        the resampled grid
     
    '''
    grid = grid[::grid_factor,::grid_factor]
    return grid
    
def truncate_image(channel, min_percentile=5, max_percentile=95):
    '''
     Remove image outliers by truncating to the defined percentiles.
     
     Inputs:
     channel :        the array to be truncated
     min_percentile : the lower bound percentile to cut at
     max_percentile : the upper bound percentile to cut at
     
     Outputs:
     channel :        the truncated array
    '''
    min_pc = np.percentile(channel[np.isfinite(channel)], min_percentile)
    max_pc = np.percentile(channel[np.isfinite(channel)], max_percentile)
    channel[channel < min_pc] = min_pc
    channel[channel > max_pc] = max_pc
    return channel
    
def norm_image(image_array, contrast=[1.0, 1.0, 1.0], unhitch=True):
    '''
     Normalise an image with either independant channels (unhitch) or 
     with combined channels.
     
     Inputs:
     image_array : the array to be normalised
     contrast :    non-linear gamma to apply
     unhitch :     switch to control normalisation by all channels (False) 
                   or channel by channel (True)
     
     Outputs:
     image_array : the normalised image
    '''
    if unhitch:
        # normalise with separating channels
        # non-linearity: contrast - note that the range is between 
        # 0 and 1, so no need to renormalise afterwards 
        for ii in range(np.shape(image_array)[-1]):
            image_array[:,:,ii] = \
                (image_array[:,:,ii] - np.nanmin(image_array[:,:,ii]))\
                / (np.nanmax(image_array[:,:,ii]) - np.nanmin(image_array[:,:,ii]))
            # apply contrast
            image_array[:,:,ii] = image_array[:,:,ii]**contrast[ii]
    else:
        # normalise without separating channels
        # non-linearity: contrast - note that the range is not between 
        # 0 and 1, so need to renormalise afterwards
        minval = np.nanmin(image_array)
        maxval = np.nanmax(image_array)
        
        for ii in range(np.shape(image_array)[-1]):
            image_array[:,:,ii] = \
                (image_array[:,:,ii] - minval)\
                / (maxval - minval)
            # apply contrast
            image_array[:,:,ii] = image_array[:,:,ii]**contrast[ii]

        minval = np.nanmin(image_array)
        maxval = np.nanmax(image_array)
        for ii in range(np.shape(image_array)[-1]):
            image_array[:,:,ii] = \
                (image_array[:,:,ii] - minval)\
                / (maxval - minval)
            
    return image_array
   
def process_image(lon, lat, red, green, blue,\
                  run_subset_image=False,\
                  subset_extents=None,\
                  run_reduce_image=False,\
                  grid_factor=5,\
                  run_truncate_image=False,\
                  min_percentile=5,\
                  max_percentile=95,\
                  contrast=[1.0,1.0,1.0],\
                  unhitch=False,\
                  run_histogram_image=False,\
                  nbins=512):
    '''
     Wrapper function for image manipulation. Calls the functions
     described above in sequence. It exists only for adding reasability
     to Jupyter Notebooks.
    '''
    
    if run_subset_image:
        i1, i2, j1, j2 = subset_image(lat, lon, subset_extents)
        lat = lat[i1:i2,j1:j2]
        lon = lon[i1:i2,j1:j2]
        red = red[i1:i2,j1:j2]
        green = green[i1:i2,j1:j2]
        blue = blue[i1:i2,j1:j2]
        
    if run_reduce_image:
        lat = reduce_image(lat, grid_factor=grid_factor)
        lon = reduce_image(lon, grid_factor=grid_factor)
        red = reduce_image(red, grid_factor=grid_factor)
        green = reduce_image(green, grid_factor=grid_factor)
        blue = reduce_image(blue, grid_factor=grid_factor)
        
    if run_truncate_image:
        red = truncate_image(red, min_percentile=min_percentile, max_percentile=max_percentile)
        green = truncate_image(green, min_percentile=min_percentile, max_percentile=max_percentile)
        blue = truncate_image(blue, min_percentile=min_percentile, max_percentile=max_percentile)
        
    height = np.shape(red)[0]
    width = np.shape(red)[1]
    image_array = np.zeros((height, width, 3), dtype=np.float32)

    image_array[..., 0] = red
    image_array[..., 1] = green
    image_array[..., 2] = blue
    image_array = norm_image(image_array, contrast=contrast, unhitch=unhitch)
    
    if run_histogram_image:
        image_array = exposure.equalize_adapthist(image_array, nbins=nbins)

    #mesh_rgb = image_array[:, :-1, :]
    mesh_rgb = image_array[:, :, :]
    colorTuple = mesh_rgb.reshape((mesh_rgb.shape[0] * mesh_rgb.shape[1]), 3)
    colorTuple = np.insert(colorTuple, 3, 1.0, axis=1)

    return image_array, colorTuple, lon, lat 

def plot_OLCI_scene(axis, lon, lat, var, run_subset_image=False, fsz=14, cmap=plt.cm.viridis,\
                    subset_extents=None, RGB_plot=False, colorTuple=None, channel_brightness=1):
    '''
     OLCI scene plotter; handles the bespoke plotting of OLCI imagery for the AI4EO MOOC
     
     Inputs:
     axis :               the axis reference to plot in
     lon :                the longitude variables
     lat :                the latitude variables
     run_subset_image :   switch to run the image subset function
     fsz :                the plot fontsize
     cmap :               the plot colour map (if not in RGB mode)
     subset_extents :     the extents to use in run_subset
     RGB_plot :           switch to determine plotting in RGB (3 channel) or data (1 channel) mode 
     colorTuple :         the array of colours to use in RGB mode
     channel_brightness : the gamma value to apply
     
     Outputs:
     plot1 : a plot handle object
    '''

    # plot the data
    if RGB_plot:
        plot1 = axis.pcolormesh(lon, lat, lon * np.nan,\
                             color=colorTuple ** channel_brightness, \
                             clip_on = True,
                             edgecolors=None, zorder=0, \
                             transform=ccrs.PlateCarree())
    else:
        plot1 = axis.pcolormesh(lon, lat, var, zorder=0, \
                             transform=ccrs.PlateCarree(), cmap=cmap)
        
    # change the plot extent if required
    if run_subset_image:
        axis.set_extent(subset_extents, crs=ccrs.PlateCarree())

    g1 = axis.gridlines(draw_labels = True, zorder=20, color='0.5', linestyle='--',linewidth=0.5)
    if run_subset_image:
        g1.xlocator = mticker.FixedLocator(np.linspace(int(subset_extents[0]), int(subset_extents[1]), 5))
        g1.ylocator = mticker.FixedLocator(np.linspace(int(subset_extents[2]), int(subset_extents[3]), 5))
    else:
        g1.xlocator = mticker.FixedLocator(np.linspace(int(np.min(lon)-1), int(np.max(lon)+1), 5))
        g1.ylocator = mticker.FixedLocator(np.linspace(int(np.min(lat)-1), int(np.max(lat)+1), 5))

    g1.xlabels_top = False
    g1.ylabels_right = False
    g1.xlabel_style = {'size': fsz, 'color': 'black'}
    g1.ylabel_style = {'size': fsz, 'color': 'black'}

    return plot1

def flag_data_fast(flags_we_want, flag_names, flag_values, flag_data, flag_type='WQSF'):
    '''
     Quick scene flagger for Sentinel-3 data. Adapted from functions developed by 
     Ben Loveday and Plymouth Marine Laboratory as part of the Copernicus Marine 
     Ocean Training Service.
     
     Inputs:
     flags_we_want : the names of the flags we want to apply
     flag_names :    all flag names
     flag_values :   all flag bit values
     flag_data :     the flag array
     flag_type :     the flag type
     
     Outputs:
     binary flag mask array
    '''
    flag_bits = np.uint64()
    if flag_type == 'SST':
        flag_bits = np.uint8()
    elif flag_type == 'WQSF_lsb':
        flag_bits = np.uint32()
    
    for flag in flags_we_want:
        try:
            flag_bits = flag_bits | flag_values[flag_names.index(flag)]
        except:
            print(flag + " not present")
    
    return (flag_data & flag_bits) > 0

def get_OLCI_RGB(input_path, run_subset_image=False,\
                 subset_extents=None,\
                 run_reduce_image=False,\
                 grid_factor=5,
                 nchannels=11,
                 return_orig_coords=False):
    '''
     Creates an RGB channel array from an OLCI L1 or L2 product
     
     Inputs:
     input_path :           the SAFE directory
     run_subset_image :     switch to subset the image channels
     subset_extents :       the i/j values of a box to extract
     run_reduce_image :     switch to resample the image a reduced grid resolution
     grid_factor:           the grid reduction parameter
     nchannels:             the number of radiomatery channels to use 
                            (defult 11 for OLCI tristim.)
     return_orig_coords :   switch to return the original lon/lat array
     
     Outputs:
     lon/lat :              the lon/lat arrays with any subsetting and/or
                            resampling applied
     red/green/blue :       the image channels
     raster_lon/raster_lat: the original grids (if requested)
    '''
    
    if 'WFR' in input_path:
        rad_type = 'reflectance'
        rad_offset = 0.05
    else:
        rad_type = 'radiance'
        rad_offset = 1.0
    
    ds1 = xr.open_dataset(os.path.join(input_path,'geo_coordinates.nc'))
    raster_lat = ds1.latitude.data
    raster_lon = ds1.longitude.data
    ds1.close()
    
    lon = raster_lon.copy()
    lat = raster_lat.copy()
    
    if not return_orig_coords:
        raster_lat = None
        raster_lon = None
    
    if run_subset_image:
        if 'int' in str(type(subset_extents[0])):
            i1 = subset_extents[0] ; i2 = subset_extents[1]
            j1 = subset_extents[2] ; j2 = subset_extents[3]
        else:
            i1, i2, j1, j2 = subset_image(raster_lat, raster_lon, subset_extents)
        lat = lat[i1:i2,j1:j2]
        lon = lon[i1:i2,j1:j2]
    
    if run_reduce_image:
        lat = reduce_image(lat,grid_factor)
        lon = reduce_image(lon,grid_factor)

    channel_dict = {}
    for rad_channel_number in range(1, nchannels+1):
        channel_name = str(rad_channel_number).zfill(2)
        rad_channel = 'Oa%s_%s' % (str(rad_channel_number).zfill(2),rad_type)
        rad_file = os.path.join(input_path, rad_channel + '.nc') 
        rad_fid = xr.open_dataset(rad_file)
        
        if run_subset_image:
            if run_reduce_image:
                channel_dict["Ch{}".format(channel_name)] = \
                  rad_fid.variables[rad_channel].data[i1:i2:grid_factor,j1:j2:grid_factor]
            else:
                channel_dict["Ch{}".format(channel_name)] = \
                  rad_fid.variables[rad_channel].data[i1:i2,j1:j2]
        else:
            if run_reduce_image:
                channel_dict["Ch{}".format(channel_name)] = \
                  rad_fid.variables[rad_channel].data[::grid_factor,::grid_factor]
            else:
                channel_dict["Ch{}".format(channel_name)] = \
                  rad_fid.variables[rad_channel].data
                
        rad_fid.close()

    # tristimulus build for RGB channels 
    red = np.log10(rad_offset \
          + 0.01 * channel_dict['Ch01'] \
          + 0.09 * channel_dict['Ch02'] \
          + 0.35 * channel_dict['Ch03'] \
          + 0.04 * channel_dict['Ch04'] \
          + 0.01 * channel_dict['Ch05'] \
          + 0.59 * channel_dict['Ch06'] \
          + 0.85 * channel_dict['Ch07'] \
          + 0.12 * channel_dict['Ch08'] \
          + 0.07 * channel_dict['Ch09'] \
          + 0.04 * channel_dict['Ch10'])
    
    green = np.log10(rad_offset \
          + 0.26 * channel_dict['Ch03'] \
          + 0.21 * channel_dict['Ch04'] \
          + 0.50 * channel_dict['Ch05'] \
          + 1.00 * channel_dict['Ch06'] \
          + 0.38 * channel_dict['Ch07'] \
          + 0.04 * channel_dict['Ch08'] \
          + 0.03 * channel_dict['Ch09'] \
          + 0.02 * channel_dict['Ch10'])
    
    blue = np.log10(rad_offset \
          + 0.07 * channel_dict['Ch01'] \
          + 0.28 * channel_dict['Ch02'] \
          + 1.77 * channel_dict['Ch03'] \
          + 0.47 * channel_dict['Ch04'] \
          + 0.16 * channel_dict['Ch05'])

    return lon, lat, red, green, blue, raster_lon, raster_lat

def add_boxes(axis, spectral_plot_cols, spectral_box_extents):
    '''
     Function to add a box to a map plot
     
     Inputs:
     axis :                 an axis handle
     spectral_plot_cols :   a list of colours to use for plotting
     spectral_box_extents : a list of points to use for the boxes
     
     Outputs:
     None
    '''
    for col,extent in zip(spectral_plot_cols, spectral_box_extents):
        axis.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],\
                  [extent[2], extent[2], extent[3], extent[3], extent[2]],
                  color=col, linewidth=1, transform=ccrs.Geodetic())
