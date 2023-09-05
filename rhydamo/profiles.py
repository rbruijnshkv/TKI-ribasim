import xarray as xr
import pandas as pd

class profile_generator:
    def __init__(self):
        pass
    
    
    
    
    def load_VolumeTool(self, VT_path):
        """
        Load a VolumeTool dataset from the specified path and store it in the current instance.

        Parameters:
        VT_path (str): The file path to the VolumeTool dataset to be loaded.

        Returns:
        xr.Dataset: The loaded VolumeTool dataset.
        """
        VT = xr.open_dataset(VT_path)
        self.VT = VT
        return VT
    
    
    
    
    def create_profile_table(self, VT, increment_VT, basins):
        """
        Creates a profile table by combining relevant parameters from the VolumeTool (VT) instance.

        Parameters:
        VT (VolumeTool): An instance of the VolumeTool class containing volume, surface, and bedlevel data.

        Returns:
        pandas.DataFrame: A DataFrame containing the combined information from the VolumeTool instance.
                          The DataFrame includes columns for 'volume', 'area', 'node_id', 'levels',
                          'bedlevel', 'level_height', and 'height'.
                          Rows with zero volume and area are filtered out.
        """
        #define relevant parameters of the VolumeTool
        volume = VT.volume.to_dataframe()
        surface = VT.surface.to_dataframe()
        bedlevel = VT.bedlevel.to_dataframe()

        #add the relevant parameters of the VolumeTool to a dataframe. 
        #For now, the profile table is called "combined"
        combined = pd.DataFrame()
        combined['volume'] = volume
        combined['area'] = surface
        combined = combined.reset_index(level=[0,1])
        combined.rename({'mesh1d_nNodes': 'node_id'}, axis=1, inplace = True)
        combined.node_id += 1
        bedlevel.index += 1
        combined = combined.merge(bedlevel, left_on = 'node_id', right_on = bedlevel.index) #add everywhere bedlevels to calculate the level per step

        #add height to the dataframe
        height = combined.bedlevel + combined.levels * increment_VT
        combined['level_height'] = height
        combined['height'] = combined.level_height - combined.bedlevel #add height compared to bedlevel

        #remove rows where both the volume and storage are 0.
        combined = combined.loc[(combined.volume != 0) & (combined.area !=0)]
        self.combined = combined
        
        return combined
    
    
    def post_process_profile_table(self, profiles, basins, terminal, fill_value = 0.001):
        """
        Post-process a profile table to remove non-ascending areas and volumes for each node,
        and filter profiles based on the specified basins.

        Parameters:
        profiles (pandas.DataFrame): The input profile table DataFrame containing columns like 'node_id',
                                     'level_height', 'area', and 'volume'.
        basins (pandas.DataFrame): The DataFrame containing basin information to filter the profiles.
                                   It should at least have a 'node_id' column.
        fill_value (float, optional): The value to add to adjusted areas in case of non-ascending values.
                                      Defaults to 0.001.

        Returns:
        pandas.DataFrame: The post-processed profile table DataFrame with non-ascending rows removed,
                          potentially adjusted areas, and filtered based on the specified basins.

        """
        combined = profiles
        combined = round(combined,5)
        combined['row_id'] = combined.index
        combined = combined.sort_values(by=['node_id', 'level_height'])

        #now remove non ascending values    
        print('- Removing non ascending areas and volumes')

        count=0
        indexes_to_remove = []    

        previous_area_value = combined.iloc[0].area
        previous_volume_value = combined.iloc[0].volume
        previous_nodeid_value = combined.iloc[0].node_id

        for i in range(len(combined)-1):
            new_area_value = combined.iloc[i+1].area
            new_volume_value = combined.iloc[i+1].volume
            new_nodeid_value = combined.iloc[i+1].node_id

            length_nodeid = len(combined.loc[combined.node_id == new_nodeid_value])
            #a counter should be created to check whether at least 2 rows remain
            if previous_nodeid_value != new_nodeid_value: #if the node_id changes, set the count back to 
                count = 0

            if (previous_area_value >= new_area_value or previous_volume_value >=new_volume_value) and previous_nodeid_value == new_nodeid_value:
                count+=1 
                row_id = combined.iloc[i+1].row_id

                if length_nodeid - count > 2: #at least two rows should remain. 
                    indexes_to_remove.append(row_id)
                else:
                    combined.loc[combined.row_id == row_id, 'area'] += fill_value #avoid double level values
                    print('test')
            else:
                previous_area_value = new_area_value
                previous_volume_value = new_volume_value
                previous_nodeid_value = new_nodeid_value

        combined = combined.loc[~combined.row_id.isin(indexes_to_remove)]  
        
        #only use profiles if a basin is present
        combined = combined[combined['node_id'].isin(basins['node_id'])]
        
        combined = combined.rename(columns = {'level_height': 'level'})
       # combined.set_index('node_id', drop = True, inplace = True)
        # combined = combined[['node_id', 'level', 'area', 'bedlevel']]
        combined = combined[['node_id', 'level', 'area', 'bedlevel']]
        
        combined = combined.loc[combined.node_id != terminal.node_id.values[0]]
        
        return combined, indexes_to_remove


    def create_IC(self, profiles, initial_height):
        """
        Create initial conditions (IC) based on the given profiles and initial height.

        Parameters:
        profiles (pandas.DataFrame): The input profile table DataFrame containing columns like 'node_id'
                                     and 'bedlevel'.
        initial_height (float): The initial height to be added to the bedlevel for creating the initial condition.

        Returns:
        pandas.DataFrame: A DataFrame containing the initial conditions for each node_id.
        """
        
        IC = profiles
        # IC['node_id'] = IC.index.values
        IC = IC.drop_duplicates(subset = 'node_id') #only get IC's for basins with a profile
        IC = IC[['node_id', 'bedlevel']]
        IC['level'] = IC['bedlevel'] + initial_height
        IC = IC[['node_id', 'level']]
        IC = IC.reset_index(drop=True)
        IC.index += 1
        
        return IC
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


