# General packages;
# All functions authored by Ben Loveday (2021) unless otherwise specified, 
# please attribute appropriately.

import glob
import os
import numpy as np
import netCDF4 as nc
import xmltodict
import xarray as xr
from rasterio.warp import transform
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

# Specific packages
from MDN import meta

def set_experiment(experiments, name, exp_dict):
    '''
     Function to create an MDN experiment
    
     Inputs:
     experiments : the master experiment dictionary
     name        : the name of this experiment
     exp_dict    : the key value pairs of this experiements parameters
     
     Outputs:
     experiments
    '''
    experiments[name] = {}
    for key in exp_dict:
        experiments[name][key] = exp_dict[key]
    return experiments

def read_S3_L2_standard(input_dir, sensor, coords=None, rrs_correct=1):
    '''
     Function to read Sentinel-3 L2 standard product into the correct format
     for ingestion into the MDN machine learning module.
     
     Inputs:
     input_dir :   input Sentinel-3 product (.SEN3 directory)
     sensor :      sensor type
     coords :      subsetting coordinates, either by pixel (integers [x1, x2, y1, y2])
                   or lat/lon (float [lon1, lat1, lon2, lat2])
     rrs_correct : scale factor for deriving Rrs from reflectance
     
     Outputs:
     Rad :         stack of radiance/reflectance arrays
     lon/lat :     coordinate grids
    '''
    # check MDN required bands
    MDN_bands = meta.SENSOR_BANDS[sensor]
    
    # check available satellite bands and band names
    xml_file = glob.glob(os.path.join(input_dir,'xfdu*'))[0]
    with open(xml_file) as fxml:
        product_metadata = xmltodict.parse(fxml.read())

    product_bands = [] ; band_names = []
    band_count = -1
    band_found = True
    
    while band_found:
        band_count = band_count + 1
        try:
            product_bands.append(float(product_metadata['xfdu:XFDU']['metadataSection']['metadataObject']\
              [4]['metadataWrap']['xmlData']['olci:olciProductInformation']\
              ['olci:bandDescriptions']['sentinel3:band'][band_count]['sentinel3:centralWavelength']))
            band_names.append(product_metadata['xfdu:XFDU']['metadataSection']['metadataObject']\
              [4]['metadataWrap']['xmlData']['olci:olciProductInformation']\
              ['olci:bandDescriptions']['sentinel3:band'][band_count]['@name'])
        except:
            band_found = False

    # Match and read bands
    rad_dict = {}
    init = True
    for MDN_band in np.array(MDN_bands).astype(float):

        if init:
            # get geo coords
            nc_fid = nc.Dataset(glob.glob(os.path.join(input_dir,'geo_coordinates.nc'))[0])
            if coords == None:
                lon = nc_fid.variables['longitude'][:]
                lat = nc_fid.variables['latitude'][:]
            else:
                lon = nc_fid.variables['longitude'][coords[0]:coords[1],coords[2]:coords[3]]
                lat = nc_fid.variables['latitude'][coords[0]:coords[1],coords[2]:coords[3]]
            init = False
            
        closest_band = np.where(np.min(abs(product_bands - MDN_band)) == abs(product_bands - MDN_band))[0][0]
        this_band = band_names[closest_band]
        print('Closest band for {MDN_band}: {band}'.format(MDN_band=str(int(MDN_band)), band=this_band))
        
        nc_fid = nc.Dataset(glob.glob(os.path.join(input_dir,this_band+'*'))[0])
        if coords == None:
            rad_dict['rrs_'+str(int(MDN_band))] = nc_fid.variables[this_band+'_reflectance'][:] / rrs_correct            
        elif 'int' in str(type(coords[0])):
            rad_dict['rrs_'+str(int(MDN_band))] = nc_fid.variables[this_band+'_reflectance']\
              [coords[0]:coords[1],coords[2]:coords[3]] / rrs_correct
        elif 'float' in str(type(coords[0])):
            rad_dict['rrs_'+str(int(MDN_band))] = nc_fid.variables[this_band+'_reflectance']\
              [coords[0]:coords[1],coords[2]:coords[3]] / rrs_correct            
        nc_fid.close()

    # stack channels
    rad_stack = []
    for item in rad_dict:
        rad_stack.append(rad_dict[item])
    Rad = np.dstack(rad_stack)
    
    return Rad, lon, lat

def read_S2_L2_standard(input_dir, sensor, coords=None, rrs_correct=1, resolution=60, mask=True):
    '''
     Function to read Sentinel-2 L2 standard product into the correct format
     for ingestion into the MDN machine learning module.

     Inputs:
     input_dir :   input Sentinel-2 product (.SAFE directory)
     sensor :      sensor type
     coords :      subsetting coordinates, either by pixel (integers [x1, x2, y1, y2])
                   or lat/lon (float [lon1, lat1, lon2, lat2])
     rrs_correct : scale factor for deriving Rrs from reflectance

     Outputs:
     Rad :         stack of radiance/reflectance arrays
     lon/lat :     coordinate grids
     '''
    # check MDN required bands
    MDN_bands = meta.SENSOR_BANDS[sensor]
    
    # check available satellite bands and band names
    xml_file = glob.glob(os.path.join(input_dir,'MTD_*.xml'))[0]
    with open(xml_file) as fxml:
        product_metadata = xmltodict.parse(fxml.read())

    product_bands = [] ; band_names = []
    band_count = -1
    band_found = True
    
    while band_found:
        band_count = band_count + 1
        try:
            product_bands.append(float(product_metadata['n1:Level-2A_User_Product']['n1:General_Info']\
              ['Product_Image_Characteristics']['Spectral_Information_List']['Spectral_Information']\
              [band_count]['Wavelength']['CENTRAL']['#text']))

            band_name = product_metadata['n1:Level-2A_User_Product']['n1:General_Info']\
              ['Product_Image_Characteristics']['Spectral_Information_List']['Spectral_Information']\
              [band_count]['@physicalBand']
            if len(band_name) == 2:
                band_name = band_name[0]+'0'+band_name[-1]
            band_names.append(band_name)
        except:
            band_found = False

    # Match and read bands
    rad_dict = {}
    init = True
    for MDN_band in np.array(MDN_bands).astype(float):
        closest_band = np.where(np.min(abs(product_bands - MDN_band)) == abs(product_bands - MDN_band))[0][0]
        this_band = band_names[closest_band]
        print('Closest band for {MDN_band}: {band}'.format(MDN_band=str(int(MDN_band)), band=this_band))
        input_file = glob.glob(os.path.join(input_dir,'GRANULE','*','IMG_DATA','R'+str(resolution)+'m','*'+this_band+'*'))[0]
        
        if init:
            # get geo coords
            spatial = xr.open_rasterio(input_file)
            ny, nx = len(spatial['y']), len(spatial['x'])
            x, y = np.meshgrid(spatial['x'], spatial['y'])
            lon, lat = transform(spatial.crs, {'init': 'EPSG:4326'},
                 x.flatten(), y.flatten())
            lon = np.asarray(lon).reshape((ny, nx))
            lat = np.asarray(lat).reshape((ny, nx))
            init = False

        if coords == None:
            rad_dict['rrs_'+str(int(MDN_band))] = np.squeeze(xr.open_rasterio(input_file).data) / rrs_correct
        elif 'int' in str(type(coords[0])):
            rad_dict['rrs_'+str(int(MDN_band))] = np.squeeze(xr.open_rasterio(input_file).data) / rrs_correct
        elif 'float' in str(type(coords[0])):
            rad_dict['rrs_'+str(int(MDN_band))] = np.squeeze(xr.open_rasterio(input_file).data) / rrs_correct
            
    # stack channels
    rad_stack = []
    for item in rad_dict:
        rad_stack.append(rad_dict[item])
    Rad = np.dstack(rad_stack)

    if mask:
        # screen Rad by the B07 value to remove land points, as Rrs in the NIR for water should be low
        mask_indices = np.where((Rad[:,:,-1] > 0.05))
        for ii in range(np.shape(Rad)[-1]):
            Rad[mask_indices[0],mask_indices[1],ii] = np.nan

    return Rad, lon, lat

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