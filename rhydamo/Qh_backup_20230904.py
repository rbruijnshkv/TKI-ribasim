import pandas as pd
import geopandas as gpd
import numpy as np
import xugrid as xu

class Qh_relations:
    def __init__(self):
        pass
    
    def create_single_Qh_relations(map_file, TRC_basin_relation, n_average_samples=3):
        """
        Creates a DataFrame of Qh relations based on data from a single '_map.nc' file.

        This function extracts discharge (Q) and water level (h) data from a given '_map.nc' file,
        associates them with basin and edge information from TRC_basin_relation, and constructs a DataFrame
        containing relevant information for Qh relations.

        Parameters:
            map_file (xarray.Dataset): The '_map.nc' file opened using xarray.
            TRC_basin_relation (pd.DataFrame): DataFrame containing the basin-edge relationship data.
            n_average_samples (int, optional): Number of samples to average for Q and h. Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame containing Qh relations along with discharge and level coordinates.
        """

        Q = np.mean(map_file['mesh1d_q1'].values[-n_average_samples:], axis=0)
        h = np.mean(map_file['mesh1d_s1'].values[-n_average_samples:], axis=0)

        temp_Qh = pd.DataFrame(columns=['discharge', 'level', 'node_id', 'discharge_coord', 'level_coord'])
        temp_Qh['node_id'] = np.arange(len(Q) + 1, len(Q) + 1 + len(h))

        from_basin_indices = TRC_basin_relation['from_basin'] - 1
        from_edge_indices = TRC_basin_relation['edge_id'] - 1
        Q_points = np.round(Q[from_edge_indices], 4)
        h_points = np.round(h[from_basin_indices], 4)

        temp_Qh['discharge'] = Q_points
        temp_Qh['level'] = h_points

        #add some coordinates for later checks    
        Q_coordx = map_file['mesh1d_q1'].coords['mesh1d_edge_x'].values
        Q_coordy = map_file['mesh1d_q1'].coords['mesh1d_edge_y'].values
        h_coordx = map_file['mesh1d_s1'].coords['mesh1d_node_x'].values
        h_coordy = map_file['mesh1d_s1'].coords['mesh1d_node_y'].values

        Q_geometry = gpd.points_from_xy(Q_coordx, Q_coordy)
        h_geometry = gpd.points_from_xy(h_coordx, h_coordy)

        temp_Qh['discharge_coord'] = Q_geometry
        temp_Qh['level_coord'] = h_geometry

        return temp_Qh#.loc[temp_Qh.node_id == 4347]
    
    
    
    def create_Qh_relations(self, nc_file_paths, TRC_basin_relation, n_average_samples):
        Qh = pd.DataFrame(columns = ['discharge', 'level', 'node_id', 'n_simulation'])

        for i in range(len(nc_file_paths)):
            map_file = xu.open_dataset(nc_file_paths[i])
            temp_Qh = Qh_relations.create_single_Qh_relations(map_file = map_file, 
                                                 TRC_basin_relation = TRC_basin_relation,
                                                 n_average_samples = n_average_samples)
            temp_Qh['n_simulation'] = i + 1 #add the simulation number, for aggregation purposes later
            Qh = pd.concat([Qh, temp_Qh])
        
        return Qh

    def add_zero_point(Qh_per_node_id):
        Qh_per_node_id.sort_values(by=['node_id', 'level'],inplace=True)
        level1, level2 = Qh_per_node_id.level.iloc[0], Qh_per_node_id.level.iloc[1] 
        discharge1, discharge2 = Qh_per_node_id.discharge.iloc[0], Qh_per_node_id.discharge.iloc[1] 
        a = (level2 - level1) / (discharge2 - discharge1)
        b = level1 - a * discharge1
        # b = level0
        return(b)

    def identify_negative_flow_directions(self, Qh, edge_nodes, minimum_threshold, revert_line = True):
        """
        Identifies and handles negative flow directions in Qh relations based on discharge values.

        This function analyzes the discharge column in the Qh DataFrame to identify nodes where negative flows
        occur frequently across simulations. It allows for the option to revert the direction of the edges in the
        edge_nodes array based on the identified negative flow nodes. Additionally, the 'discharge' values
        for the identified negative flow nodes are multiplied by -1 to handle the inconsistency in flow direction.

        Parameters:
            Qh: DataFrame containing Qh relations.
            edge_nodes: Array representing edge-node relationships.
            revert_line: Whether to revert the direction of drawing edges. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - Qh: Updated Qh DataFrame with adjusted discharge values.
                - number_of_negative_flows: Series containing node_ids with frequent negative flows.
                - edge_nodes: Updated edge_nodes array if the direction is reverted.
        """

        grouped = Qh.groupby('node_id')['discharge'].apply(lambda x: (x < minimum_threshold).sum()) # Group the DataFrame by the node_id and count the number of negative discharge values    
        threshold = round(Qh.n_simulation.max() / 2) #determine the threshold value when the majority of the times the flow reverses compared to how the edge has been drawn
        number_of_negative_flows = grouped[grouped>threshold]
        
        if revert_line == True: #revert the direction of drawing of the edge. This is of importance later when the network is constructed
            index_numbers = number_of_negative_flows.index.values # Get the index numbers from number_of_negative_flows
            edge_nodes_swapped = np.copy(edge_nodes) # Create a copy of edge_nodes
            edge_nodes_swapped[index_numbers, 0], edge_nodes_swapped[index_numbers, 1] = edge_nodes[index_numbers, 1], edge_nodes[index_numbers, 0] # Swap the values at the specified rows
            edge_nodes = edge_nodes_swapped # Place the modified edge_nodes_swapped array back into edge_nodes

        #Multiply the discharge of the found node_id's with minus one. Note that this does not prevent some discharge to be negative, as it only multiplies the discharge when the majority of the simulations have negative flows
        mask = Qh['node_id'].isin(number_of_negative_flows.index)
        Qh.loc[mask, 'discharge'] *= -1 # Multiply the 'discharge' values by -1 for the matching rows

        return Qh, number_of_negative_flows, edge_nodes
    
    def Qh_post_process(self, Qh, minimum_threshold, fill_value):
        #to do: efficienter maken
        
        '''Post process the Qh relations node per node'''
        print('- Removing too low discharges')
        #remove rows where the flow is too small, or where duplicates occur.
        Qh = Qh.dropna() #remove NaNs
        Qh = Qh.sort_values(by=['node_id', 'level', 'discharge', 'n_simulation'])
        Qh.reset_index(drop=True, inplace = True)

        Qh['level'] = Qh['level'].round(4)
        Qh['discharge'] = Qh['discharge'].round(4)
        Qh['row_id'] = Qh.index
        indexes_to_remove = []
        alterated_Qh_index = []
        #remove all values for the discharge lower than the threshold. Keep at least two values, as this is required by Ribasimj.
        for count in range(len(Qh.node_id.unique())): #loop through each unique Qh relation
            i = Qh.node_id.unique()[count]
            sample = Qh.loc[Qh.node_id == i]

            if len(sample[sample.discharge > minimum_threshold]) == 0: 
                sorted_sample = sample[sample.discharge < minimum_threshold].sort_values(by='discharge', ascending=False) #only select in a new df the rows where the discharge is lower than the threshold, sort it
                top_discharges = sorted_sample.head(2)
                sample.loc[Qh['row_id'].isin(top_discharges.index.values), 'discharge'] = fill_value
                Qh.loc[Qh['row_id'] == top_discharges.index.values[1], 'discharge'] = 2 * fill_value #time two to avoid the same Q
                Qh.loc[Qh['row_id'] == top_discharges.index.values[0], 'discharge'] = fill_value                

                if abs(top_discharges.iloc[1].level < top_discharges.iloc[0].level) < fill_value:
                    Qh.loc[Qh['row_id'] == top_discharges.index.values[1], 'level'] += fill_value

                alterated_Qh_index.append(i)

            if len(sample[sample.discharge > minimum_threshold]) == 1:
                sorted_sample = sample.sort_values(by='discharge', ascending=False) #only select in a new df the rows where the discharge is lower than the threshold, sort it
                top_discharges = sorted_sample.iloc[1, :]   
                sample.loc[sample['row_id'] == top_discharges.row_id, 'discharge'] = sorted_sample.iloc[0].discharge + fill_value #update in sample for later post processing
                sample.loc[sample['row_id'] == top_discharges.row_id, 'level'] = sorted_sample.iloc[0].level + fill_value #otherwise the level is the same, which will be filtered out later #update in sample for later post processing 
                Qh.loc[Qh['row_id'] == top_discharges.row_id, 'level'] = sorted_sample.iloc[0].level + fill_value #otherwise the level is the same, which will be filtered out later #update in sample for later post processing
                Qh.loc[Qh['row_id'] == top_discharges.row_id, 'discharge'] = sorted_sample.iloc[0].discharge + fill_value #update in the original df
                alterated_Qh_index.append(i)

            rows_to_remove = sample.loc[sample.discharge < minimum_threshold].index.values

            if len(rows_to_remove)>0: 
                indexes_to_remove.append(rows_to_remove)


        indexes_to_remove = np.concatenate(indexes_to_remove) #flatten
        Qh = Qh.loc[~Qh.row_id.isin(indexes_to_remove)]    


    #now remove non ascending values    
        print('- Removing non ascending discharges and levels')

        indexes_to_remove2 = []    

        Qh = Qh.sort_values(by=['node_id', 'level', 'discharge'])
        previous_discharge_value = Qh.iloc[0].discharge
        previous_level_value = Qh.iloc[0].level
        previous_nodeid_value = Qh.iloc[0].node_id

        for i in range(len(Qh)-1):
            new_discharge_value = Qh.iloc[i+1].discharge
            new_level_value = Qh.iloc[i+1].level
            new_nodeid_value = Qh.iloc[i+1].node_id

            length_nodeid = len(Qh.loc[Qh.node_id == new_nodeid_value])
            #a counter should be created to check whether at least 2 rows remain
            if previous_nodeid_value != new_nodeid_value: #if the node_id changes, set the count back to 0
                count = 0

            if (previous_discharge_value >= new_discharge_value or previous_level_value >=new_level_value) and previous_nodeid_value == new_nodeid_value:
                count+=1 
                row_id = Qh.iloc[i+1].row_id

                if length_nodeid - count > 2: #at least two rows should remain. 
                    indexes_to_remove2.append(row_id)
                else:
                    Qh.loc[Qh.row_id == row_id, 'level'] = previous_level_value + fill_value #avoid double level values
                    Qh.loc[Qh.row_id == row_id, 'discharge'] = previous_discharge_value + fill_value #avoid double level values
            else:
                previous_discharge_value = new_discharge_value
                previous_level_value = new_level_value
                previous_nodeid_value = new_nodeid_value

        Qh = Qh.loc[~Qh.row_id.isin(indexes_to_remove2)]   


        #add zero point
        print('- Add extrapolation points to Q = 0')
        Qh.sort_values(by=['node_id', 'level'],inplace=True)
        Qh.reset_index(drop=True, inplace=True)
        Qh_new = Qh.copy(deep=True)

        Qh_zeros = pd.DataFrame(columns=Qh_new.columns)
        Qh_zeros['node_id'] = Qh.node_id.unique()
        Qh_zeros['discharge'], Qh_zeros['n_simulation'] = 0, 0 

        for i in Qh.node_id.unique():
            Qh_per_node_id = Qh.loc[Qh.node_id == i].copy()
            new_point = Qh_relations.add_zero_point(Qh_per_node_id) #level at which Q = 0
            Qh_zeros.loc[Qh_zeros.node_id == i, 'level'] = new_point
        Qh_new = pd.concat([Qh_new, Qh_zeros])
        Qh_new.sort_values(by=['node_id', 'level'],inplace=True)

        Qh = Qh_new

        return Qh, (indexes_to_remove, indexes_to_remove2), alterated_Qh_index

    
    def set_minimum_dt_Qh(self, Qh, dh_Qh_relation, changed_Qh_index):
        #to do 2207 eruit halen
        Qh_post_processed = Qh
        alterated_Qh_index = changed_Qh_index
        
        lowest_Qh_post_processed = Qh_post_processed.drop_duplicates(keep='first', subset = 'node_id').reset_index(drop=True)
        highest_Qh_post_processed = Qh_post_processed.drop_duplicates(keep='last', subset = 'node_id').reset_index(drop=True)

        Delta_h = highest_Qh_post_processed['level'] - lowest_Qh_post_processed['level']
        Delta_h_low = Delta_h[Delta_h<dh_Qh_relation]
        Delta_h_low_index = Delta_h_low.index.values
        Delta_h_low_index = Delta_h_low_index[Delta_h_low_index!=2207] #avoid the edge to the terminal becoming a manning resistance

        Delta_h_low_node_id = highest_Qh_post_processed.iloc[Delta_h_low_index].node_id

        for node_id in Delta_h_low_node_id:
            matching_rows = Qh_post_processed[Qh_post_processed['node_id'] == node_id]
            if not matching_rows.empty:
                last_index = matching_rows.index[-1]
                Qh_post_processed.at[last_index, 'level'] += dh_Qh_relation
                
                Qh_post_processed = Qh
                        
        alterated_Qh = Qh.iloc[alterated_Qh_index]
        Qh_post_processed_plot = gpd.GeoDataFrame(Qh_post_processed, geometry = 'level_coord')
        alterated_Qh = gpd.GeoDataFrame(alterated_Qh, geometry = 'level_coord')

        new_manning_nodes = Qh_post_processed.loc[Qh_post_processed.node_id.isin(alterated_Qh_index)]
        new_manning_nodes = gpd.GeoDataFrame(new_manning_nodes, geometry = 'level_coord')

        # #add the new_manning_nodes to the altarated Qh
        # alterated_Qh = pd.concat([alterated_Qh, new_manning_nodes])
        # alterated_Qh.drop_duplicates(inplace=True)
        
                
        return Qh_post_processed, new_manning_nodes, alterated_Qh
    
    def last_check_Qh(self, Qh, fill_value):
        print('- Perform the last check on the Qh relations.')
        fill_value = fill_value
        Qh_post_processed = Qh
        
        indexes_to_remove2 = []    

        Qh_post_processed = Qh_post_processed.sort_values(by=['node_id', 'level', 'discharge'])
        previous_discharge_value = Qh_post_processed.iloc[0].discharge
        previous_level_value = Qh_post_processed.iloc[0].level
        previous_nodeid_value = Qh_post_processed.iloc[0].node_id

        for i in range(len(Qh_post_processed)-1):
            new_discharge_value = Qh_post_processed.iloc[i+1].discharge
            new_level_value = Qh_post_processed.iloc[i+1].level
            new_nodeid_value = Qh_post_processed.iloc[i+1].node_id

            length_nodeid = len(Qh_post_processed.loc[Qh_post_processed.node_id == new_nodeid_value])
            #a counter should be created to check whether at least 2 rows remain
            if previous_nodeid_value != new_nodeid_value: #if the node_id changes, set the count back to 0
                count = 0

            if (previous_discharge_value >= new_discharge_value or previous_level_value >=new_level_value) and previous_nodeid_value == new_nodeid_value:
                count+=1 
                row_id = Qh_post_processed.iloc[i+1].row_id

                if length_nodeid - count > 2: #at least two rows should remain. 
                    indexes_to_remove2.append(row_id)
                else:
                    Qh_post_processed.loc[Qh_post_processed.row_id == row_id, 'level'] += fill_value #avoid double level values
            else:
                previous_discharge_value = new_discharge_value
                previous_level_value = new_level_value
                previous_nodeid_value = new_nodeid_value


        Qh_post_processed = Qh_post_processed.loc[~Qh_post_processed.row_id.isin(indexes_to_remove2)] 
        
        return Qh_post_processed
    