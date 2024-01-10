from glob import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import geopy.distance

class AIS:
    ''' Class to handle AIS data and ship registry data - select vessels and plot AIS data, shift AIS data based on wind data '''

    def __init__(self, ship_registry_file, ais_folder_path, day_of_june2019, TROPOMI_flyover_unix):
        self.ship_registry_file = ship_registry_file # path to ship registry
        self.ais_folder_path = ais_folder_path # path to ais data; must be a folder with only ais data csv files

        self.day_of_june2019 = day_of_june2019 # day of June 2019 to import AIS data for
        self.TROPOMI_flyover_unix = TROPOMI_flyover_unix # unix timestamp of TROPOMI flyover

        self.ship_registry = self.ImportShipRegistry()
        self.AIS_rawdata = self.ImportAIS()

    def ImportAIS(self):
        ''' Import AIS data for a given day of June 2019 '''

        AIS_fns = sorted(glob(self.ais_folder_path+'*.csv'))
        AIS_rawdata = pd.read_csv(AIS_fns[self.day_of_june2019-1]) # assumes folder contains AIS data for all days of June 2019 in order (1-30)

        return AIS_rawdata

    def ImportShipRegistry(self):
        ''' Import ship registry '''

        ship_registry = pd.read_csv(self.ship_registry_file)

        return ship_registry
    
    def GetVessels(self, matchingshipregistry = False):
        ''' Get a list of MMSI numbers of vessels that are in the AIS data '''

        if matchingshipregistry:
            # Get the MMSI numbers of vessels that are in the AIS data and in the ship registry
            vessels = np.intersect1d(np.unique(self.AIS_rawdata['MMSI']),self.ship_registry['MMSI'])
        else:
            # Get the MMSI numbers of vessels that are in the AIS data
            vessels = np.unique(self.AIS_rawdata['MMSI'])

        return vessels

    def MergeData(self, vessels):
        ''' Make a dictionary with the metadata of each vessel and all the data from the timestamps within 3 hours before TROPOMI flies over
            Input:
                - vessels: list of MMSI numbers of vessels to include
                - TROPOMI_flyover_unix: unix timestamp of TROPOMI flyover
            Output (as attributes of AIS class):
                - AIS_data: dictionary with data of each vessel
                - AIS_metadata: dictionary with metadata of each vessel
        '''

        ship_registry = self.ship_registry
        AIS_rawdata = self.AIS_rawdata

        ship_registry_keys = ship_registry.keys()
        AIS_all_keys = AIS_rawdata.keys()
        AIS_data_keys = ['Timestamp','Heading','Speed','Latitude','longitude']

        # create a metadata dictionary with the ordered metadata of all ships
        # Layout: AIS_metadata['data type'] = 'ordered list of datatypes for all ships'
        AIS_metadata = {'MMSI':vessels}
        for key in AIS_all_keys:
            if key not in AIS_data_keys:
                AIS_metadata[key] = []
        for key in ship_registry_keys:
                AIS_metadata[key] = []

        # Create a data dictionary with the data (and metadata) of each ship
        # Layout: AIS_data['MMSI number of ship']['data type'] = \
        #         'array with the data (e.g. longitude or speed) or meta-data entry'
        AIS_data = {}
        for ves in vessels:
            AIS_data[ves] = {}
            ves_msk = AIS_rawdata['MMSI'] == ves
            
            for key in AIS_all_keys:
                # For the metadata we only need one entry
                if key not in AIS_data_keys:
                    AIS_data[ves][key] = np.array(AIS_rawdata[key][ves_msk])[0]
                    AIS_metadata[key].append(AIS_data[ves][key])
                    
                # For the data we need all rows
                else:
                    AIS_data[ves][key] = np.array(AIS_rawdata[key][ves_msk])
            # Also add the timestamp in unix (seconds after 01-01-1970 00:00 UTC)
            AIS_data[ves]['Timestamp_unix'] = np.zeros(len(AIS_data[ves]['Timestamp']),dtype=int)
            for i,time_str in enumerate(AIS_data[ves]['Timestamp']):
                AIS_data[ves]['Timestamp_unix'][i] = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ').timestamp()
            
            # Also add the metadata from the ship registry
            registry_idx = np.where(ship_registry['MMSI'] == ves)[0]
            for key in ship_registry_keys:
                if len(registry_idx) != 0:
                    AIS_data[ves][key] = ship_registry[key][registry_idx[0]]
                else:
                    AIS_data[ves][key] = np.nan
                AIS_metadata[key].append(AIS_data[ves][key])

        for key in AIS_metadata.keys():
            AIS_metadata[key] = np.array(AIS_metadata[key])

        AIS_data_keys.append('Timestamp_unix')
        # Mask the AIS data to only include the three hours before TROPOMI flies over
        for ves in vessels:
            # no later than flyover (+5 minutes measuring time)
            flyover_mask = AIS_data[ves]['Timestamp_unix'] < (self.TROPOMI_flyover_unix + (5*60))
            # no earlier than 3 hours before flyover
            flyover_mask *= AIS_data[ves]['Timestamp_unix'] > (self.TROPOMI_flyover_unix - (60*60*3))
            for key in AIS_data_keys:
                AIS_data[ves][key] = AIS_data[ves][key][flyover_mask]

        self.AIS_data = AIS_data
        self.AIS_metadata = AIS_metadata 
    
    def FilterVesselProperties(self, vessels, **kwargs):
        ''' Filter vessels based on the metadata. Always filter on time (3 hours before TROPOMI flies over) based on MergeData()
            Input:
                - vessels: list of MMSI numbers of vessels to consider
                - kwargs: key-value pairs of metadata to filter on. 
                    Key is the metadata type, value is the minimum value
            Output:
                - selec_vessels: list of MMSI numbers of selected vessels 
        '''

        # Create a mask to filter vessels
        selec_vessels_msk = np.ones(len(vessels), dtype=bool) # all vessels are selected by default

        if 'filters' in kwargs.keys():
            filters = kwargs['filters']
            # Apply filters based on user input key-value pairs
            for max_or_min, kwarg in filters.items():
                if max_or_min == 'minimum':
                    for metadata_type, value in kwarg.items():
                        selec_vessels_msk &= self.AIS_metadata[metadata_type] > value
                elif max_or_min == 'maximum':
                    for metadata_type, value in kwarg.items():
                        selec_vessels_msk &= self.AIS_metadata[metadata_type] < value
                else:
                    raise ValueError(f"Unknown key {max_or_min} in kwargs. Should be 'minimum' or 'maximum'.")
                # if values are nan, always keep them (the vessel's properties are not in the ship registry)
                selec_vessels_msk |= np.isnan(self.AIS_metadata[metadata_type])

        # Select AIS data that is at most 3 hours before TROPOMI flies over. Assumes that the AIS data is already filtered on time in MergeData()
        for i, ves in zip(np.arange(len(vessels))[selec_vessels_msk],vessels[selec_vessels_msk]):
            if len(self.AIS_data[ves]['Timestamp']) == 0:
                selec_vessels_msk[i] = False
        
        # Filter vessels based on the mask
        selec_vessels = vessels[selec_vessels_msk]

        return selec_vessels
    
    def FilterVesselLocations(self, vessels, lon_min, lon_max, lat_min, lat_max, use_any = False):
        ''' Filter vessels based on longitude, latitude. Include vessels that have at least one data point within the given range. '''

        # Create a mask to filter vessels
        selec_vessels_msk = np.ones(len(vessels), dtype=bool) # all vessels are selected by default

        # use shifted data if available
        if 'longitude_shifted' in self.AIS_data[vessels[0]].keys():
            lon_key = 'longitude_shifted'
            lat_key = 'Latitude_shifted'
        else:
            lon_key = 'longitude'
            lat_key = 'Latitude'

        for i, ves in enumerate(vessels): # at least one of lon,lat should be in range
            if use_any:   
                selec_vessels_msk[i] &= (np.any(self.AIS_data[ves][lon_key] > lon_min) & np.any(self.AIS_data[ves][lon_key] < lon_max)) & \
                                    (np.any(self.AIS_data[ves][lat_key] > lat_min) & np.any(self.AIS_data[ves][lat_key] < lat_max))
            else:
                selec_vessels_msk[i] &= (np.all(self.AIS_data[ves][lon_key] > lon_min) & np.all(self.AIS_data[ves][lon_key] < lon_max)) & \
                                    (np.all(self.AIS_data[ves][lat_key] > lat_min) & np.all(self.AIS_data[ves][lat_key] < lat_max))

        # Filter vessels based on the mask
        selec_vessels = vessels[selec_vessels_msk]
        return selec_vessels

    def PlotAISData(self, selec_vessels, ax, set_limits=False, plot_shifted=True):
        ''' Plot the AIS data of selected vessels on a map
            Input:
                - AIS_data: dictionary with data of each vessel
                - selec_vessels: list of MMSI numbers of selected vessels 
                - ax: axis to plot on
        '''	

        for i,ves in enumerate(selec_vessels):
            # use colormap to get gradient for each vessel
            colors = plt.cm.Oranges(np.linspace(0,1,len(self.AIS_data[ves]['Timestamp'])))
            N = len(self.AIS_data[ves]['longitude'])
            ax.scatter(self.AIS_data[ves]['longitude'],self.AIS_data[ves]['Latitude'],alpha=np.linspace(0.2,1,N),marker='.',color=colors,s=150)
            ax.scatter(self.AIS_data[ves]['longitude'][-1],self.AIS_data[ves]['Latitude'][-1],label=f'AIS track',marker='.', color=colors[-1],s=150)
            if 'longitude_shifted' in self.AIS_data[ves].keys() and plot_shifted: # plot shifted data if available
                ax.scatter(self.AIS_data[ves]['longitude_shifted'],self.AIS_data[ves]['Latitude_shifted'],alpha=np.linspace(0.2,1,N),marker='x',color='black',s=100)
                ax.scatter(self.AIS_data[ves]['longitude_shifted'][-1],self.AIS_data[ves]['Latitude_shifted'][-1],label=f'Wind-shifted track',marker='x', color='black',s=100)

        # set legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[0:2], labels[0:2], loc=2,bbox_to_anchor=(1,1),fontsize=14)
        ax.legend(handles[0:2], labels[0:2], loc='upper right',fontsize=14, markerscale=1.5)

        if set_limits:
            ax.set_xlim([np.min(self.AIS_data[ves]['longitude_shifted'])-0.5,np.max(self.AIS_data[ves]['longitude_shifted'])+0.5])
            ax.set_ylim([np.min(self.AIS_data[ves]['Latitude_shifted'])-0.5,np.max(self.AIS_data[ves]['Latitude_shifted'])+0.5])


    def ShiftAIS(self, northwind, eastwind, vessels):
        ''' Shift AIS ship track based on wind data to retrieve position of plume.
        '''
        
        for ves in vessels:
            # get wind data at the location of the vessel (don't use multiple wind data points because then shift will be discontinuous)
            # alternative would be to interpolate the wind data to the location of the vessel at each timestep
            alternative = True
            if alternative:
                # interpolate wind data to the location of the vessel at each timestep
                # need to use .copy because the original wind data is read-only
                northwind_nearship = np.diag( northwind.interp(longitude=self.AIS_data[ves]['longitude'],latitude=self.AIS_data[ves]['Latitude']).data[0,:,:] ).copy()
                eastwind_nearship = np.diag( eastwind.interp(longitude=self.AIS_data[ves]['longitude'],latitude=self.AIS_data[ves]['Latitude']).data[0,:,:] ).copy()

            else:
                # use the nearest wind data point
                northwind_nearship = northwind.sel(longitude=self.AIS_data[ves]['longitude'][-1],latitude=self.AIS_data[ves]['Latitude'][-1],method='nearest').data
                eastwind_nearship = eastwind.sel(longitude=self.AIS_data[ves]['longitude'][-1],latitude=self.AIS_data[ves]['Latitude'][-1],method='nearest').data

            # if there are non-nan values in the wind array, replace nans with average of wind array
            if np.sum(~np.isnan(northwind_nearship)) > 0 and np.sum(~np.isnan(eastwind_nearship)) > 0:
                northwind_nearship[np.isnan(northwind_nearship)] = np.nanmean(northwind_nearship)
                eastwind_nearship[np.isnan(eastwind_nearship)] = np.nanmean(eastwind_nearship)

            # if there are only nans in the wind array, set wind to NaN and continue to next vessel
            if np.isnan(northwind_nearship).sum() > 0 or np.isnan(eastwind_nearship).sum() > 0:
                self.AIS_data[ves]['longitude_shifted'] = np.zeros(len(self.AIS_data[ves]['longitude'])) * np.nan
                self.AIS_data[ves]['Latitude_shifted'] = np.zeros(len(self.AIS_data[ves]['Latitude'])) * np.nan
                continue
                
            # get the shift in north and east direction for each timestep
            delta_time = self.TROPOMI_flyover_unix - self.AIS_data[ves]['Timestamp_unix'] # time in seconds since TROPOMI flyover
            shift_north = northwind_nearship * delta_time # in meters
            shift_east = eastwind_nearship * delta_time # in meters

            # shift the vessel track in coordinates based on the distance shifts
            coords_tuples = list(zip(self.AIS_data[ves]['Latitude'],self.AIS_data[ves]['longitude']))
            new_coords = []
            for i,coords in enumerate(coords_tuples):
                new_coords_north = geopy.distance.distance(meters=shift_north[i]).destination(coords, bearing=0)
                new_coords.append(geopy.distance.distance(meters=shift_east[i]).destination(new_coords_north, bearing=90))
                
            # save the shifted coordinates to AIS dict
            self.AIS_data[ves]['longitude_shifted'] = np.array( [coord[1] for coord in new_coords] )
            self.AIS_data[ves]['Latitude_shifted'] = np.array( [coord[0] for coord in new_coords] )

    def GetTrackVector(self, filtered_vessels, lat_min, lat_max, lon_min, lon_max):
        ''' Get the track vectors of vessels with full tracks in given latitude, longitude range
            Input:
                - args: latitude_min, max and longitude_min, max
            Output:
                - track_vector_list: list of track vectors of vessels
        '''

        track_vector_dict = {}

        for vessel in filtered_vessels:
            if 'longitude_shifted' in self.AIS_data[vessel].keys():
                track_vector = np.array([self.AIS_data[vessel]['longitude_shifted'][-1] - self.AIS_data[vessel]['longitude_shifted'][0],
                                        self.AIS_data[vessel]['Latitude_shifted'][-1] - self.AIS_data[vessel]['Latitude_shifted'][0]])
            else:
                track_vector = np.array([self.AIS_data[vessel]['longitude'][-1] - self.AIS_data[vessel]['longitude'][0],
                                        self.AIS_data[vessel]['Latitude'][-1] - self.AIS_data[vessel]['Latitude'][0]])
            track_vector_dict[vessel] = track_vector
            
        return track_vector_dict
    
    def GetAvgShipSpeed(self, vessel):
        ''' Get the average speed of the combination of the vessel and wind '''
        distance = geopy.distance.geodesic((self.AIS_data[vessel]['Latitude_shifted'][0],self.AIS_data[vessel]['longitude_shifted'][0]),
                                             (self.AIS_data[vessel]['Latitude_shifted'][-1],self.AIS_data[vessel]['longitude_shifted'][-1])).meters
        time = self.AIS_data[vessel]['Timestamp_unix'][-1] - self.AIS_data[vessel]['Timestamp_unix'][0]
        speed = distance / time
        return speed # m/s

    def GetAvgShipSpeed_noshift(self, vessel):
        ''' Get the average speed of the combination of the vessel and wind '''
        distance = geopy.distance.geodesic((self.AIS_data[vessel]['Latitude'][0],self.AIS_data[vessel]['longitude'][0]),
                                             (self.AIS_data[vessel]['Latitude'][-1],self.AIS_data[vessel]['longitude'][-1])).meters
        time = self.AIS_data[vessel]['Timestamp_unix'][-1] - self.AIS_data[vessel]['Timestamp_unix'][0]
        speed = distance / time
        return speed # m/s
    def GetAvgShiftedShipSpeed(self, vessel):
        ''' Get the average speed of a vessel '''
        distance = geopy.distance.geodesic((self.AIS_data[vessel]['Latitude_shifted'][0],self.AIS_data[vessel]['longitude_shifted'][0]),
                                             (self.AIS_data[vessel]['Latitude_shifted'][-1],self.AIS_data[vessel]['longitude_shifted'][-1])).meters
        time = self.AIS_data[vessel]['Timestamp_unix'][-1] - self.AIS_data[vessel]['Timestamp_unix'][0]
        speed = distance / time
        return speed # m/s