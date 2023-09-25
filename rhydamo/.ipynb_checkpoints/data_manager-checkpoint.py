import os
import re 
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point, LineString, MultiPoint, MultiLineString


class NetworkProcessor:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.nc_file_paths = self.find_nc_files()

        
    def find_nc_files(self, endswith='_map.nc'):
        """
        Iterates over all subdirectories in the given model_directory and collects paths of each '_map.nc' file found.
        
        Parameters:
            endswith (str, optional): The suffix to identify target files. Defaults to '_map.nc'.
            
        Returns:
            list: A list of file paths pointing to '_map.nc' files found in the specified directory and its subdirectories.
        """
        nc_files = []
        for root, dirs, files in os.walk(self.model_directory):
            for file in files:
                if file.endswith(endswith): # find each _map.nc file in the directory
                    file_path = os.path.join(root, file)
                    nc_files.append(file_path)

        return nc_files

    
    def create_relationship_table(self, single_nc_path):
        """
        Process the data from the NC files and generate the TRC-basin relationship table. 
        Only one nc file is sufficient, as the network of the nc's are the same.
        
        Returns:
            pd.DataFrame: The TRC-basin relationship table.
        """
        map_file = xr.open_dataset(single_nc_path)
        edge_nodes = map_file['mesh1d_edge_nodes']
        
        self.map_file = map_file
        self.edge_nodes = edge_nodes
        self.single_nc_path = single_nc_path
        
        edges_nodes_df = pd.DataFrame(columns=['from_basin', 'to_basin', 'from_coord', 'to_coord', 'mid_coord', 'node_id'])
        edges_nodes_df.node_id = np.arange(1, len(edge_nodes)+1)
        edges_nodes_df.from_basin = np.array(edge_nodes)[:, 0] - 1
        edges_nodes_df.to_basin = np.array(edge_nodes)[:, 1] - 1

        for i in range(len(edge_nodes)):
            start_node = edges_nodes_df.loc[i, 'from_basin'] 
            end_node = edges_nodes_df.loc[i, 'to_basin'] 

            #retrieve from coordinates
            from_coord_x = map_file.mesh1d_s1.coords['mesh1d_node_x'].values[start_node]
            from_coord_y = map_file.mesh1d_s1.coords['mesh1d_node_y'].values[start_node]
            from_coord = gpd.GeoDataFrame(geometry = gpd.points_from_xy([from_coord_x], [from_coord_y]))

            #retrieve to coordinates
            to_coord_x = map_file.mesh1d_s1.coords['mesh1d_node_x'].values[end_node]
            to_coord_y = map_file.mesh1d_s1.coords['mesh1d_node_y'].values[end_node]
            to_coord = gpd.GeoDataFrame(geometry = gpd.points_from_xy([to_coord_x], [to_coord_y]))

            #determine mid point. This will be the location of the TRC
            mid_coord = gpd.GeoDataFrame(geometry = gpd.points_from_xy([(from_coord_x + to_coord_x)/2], [(from_coord_y + to_coord_y)/2]))

            #add the found values to the df
            edges_nodes_df.loc[i, 'from_coord'] = from_coord.values[0][0]
            edges_nodes_df.loc[i, 'to_coord'] = to_coord.values[0][0]
            edges_nodes_df.loc[i, 'mid_coord'] = mid_coord.values[0][0]
    

        edges_nodes_df.from_basin += 1
        edges_nodes_df.to_basin += 1

        mid_coord_edges = gpd.GeoDataFrame(geometry=gpd.points_from_xy(map_file['mesh1d_u1'].coords['mesh1d_edge_x'],
                                                                       map_file['mesh1d_u1'].coords['mesh1d_edge_y']))
        mid_coord_edges['edge_id'] = np.arange(1, len(mid_coord_edges)+1)
        TRC_basin_relation = edges_nodes_df.merge(right=mid_coord_edges, left_on='mid_coord', right_on='geometry')
        
        return TRC_basin_relation


    
    def attributes_from_relation(self):
        """
        Convert the coordinates of all calculation points to Basin Nodes.

        Parameters:
            x_coordinates_nodes (array-like): X coordinates of the basin nodes.
            y_coordinates_nodes (array-like): Y coordinates of the basin nodes.
            edges (list of tuples): List of tuples representing edges.

        Returns:
            tuple: A tuple containing the TRC GeoDataFrame, Basins GeoDataFrame, and Lines GeoDataFrame.
        """

        x_coordinates_nodes = self.map_file['mesh1d_node_x']
        y_coordinates_nodes = self.map_file['mesh1d_node_y']
        edges = self.edge_nodes

        #coordinates of the basin
        basins = pd.DataFrame()
        basins['x_coordinate'], basins['y_coordinate'] = x_coordinates_nodes, y_coordinates_nodes
        basins = gpd.GeoDataFrame(basins, geometry = gpd.points_from_xy(basins['x_coordinate'], basins['y_coordinate']))

        basins['NAME_ID'] = ['basin_{}'.format(i) for i in range(1, len(basins) + 1)] #naam geven
        basins['node_id'] = np.arange(1, len(basins) + 1) #vergelijkbaar als de naam, maar nu een uniek int node_id
        basins['type'] = 'Basin'
        # basins.set_index('NAME_ID', inplace=True)

        #coordinates of the TabulatedRatingCurve. Should be, for now, in the middle of two nodes
        TRC = pd.DataFrame(columns = ['x_s_node', 'y_s_node', 'x_e_node', 'y_e_node', 'x_TRC', 'y_TRC', 'from_node', 'to_node', 'type'], index = range(len(edges))) #x & y location of Starting and End node
        edges = np.array(edges)
        x_coordinates_nodes = np.array(x_coordinates_nodes)
        y_coordinates_nodes = np.array(y_coordinates_nodes)

        for i in range(len(edges)):
            start_node = np.array(edges[i])[0] - 1 #minus one since it starts counting at 1 instead of 0
            end_node = np.array(edges[i])[1] - 1 #minus one since it starts counting at 1 instead of 0  

            #filling the TabulatedRatingCurve
            TRC.loc[i, 'x_s_node'], TRC.loc[i, 'y_s_node'] = x_coordinates_nodes[start_node], y_coordinates_nodes[start_node] #filling coordinates of starting node
            TRC.loc[i, 'x_e_node'], TRC.loc[i, 'y_e_node'] = x_coordinates_nodes[end_node], y_coordinates_nodes[end_node] #filling coordinates of ending nodes

            TRC.loc[i, 'from_node'], TRC.loc[i, 'to_node'] = basins.NAME_ID.iloc[start_node], basins.NAME_ID.iloc[end_node]

        TRC['x_TRC'] = (TRC['x_s_node'] + TRC['x_e_node']) / 2
        TRC['y_TRC'] = (TRC['y_s_node'] + TRC['y_e_node']) / 2
        TRC['type'] = 'TabulatedRatingCurve'
        TRC['NAME_ID'] = ['TRC_{}'.format(i) for i in range(1, len(TRC) + 1)] #naam geven
        TRC['node_id'] = np.arange(1 + len(basins), len(TRC) + len(basins) + 1) #vergelijkbaar als de naam, maar nu een uniek int node_id. len(basins) erbij om unieke waardes te houden
        TRC['node_id_new'] = TRC['from_node'].str.split('_').str.get(1) #new

        TRC = gpd.GeoDataFrame(TRC, geometry = gpd.points_from_xy(TRC['x_TRC'], TRC['y_TRC']))

        #lines
        # Create a new GeoDataFrame
        lines = gpd.GeoDataFrame()

        # Create start and end points columns in the GeoDataFrame
        lines['start'] = gpd.points_from_xy(TRC['x_s_node'], TRC['y_s_node'])
        lines['end'] = gpd.points_from_xy(TRC['x_e_node'], TRC['y_e_node'])
        lines['from_node'], lines['to_node'] = TRC['from_node'], TRC['to_node']

        # Create lines by combining the start and end points
        lines['geometry'] = [LineString([start, end]) for start, end in zip(lines['start'], lines['end'])]
        lines = lines[['from_node', 'to_node', 'geometry']]


        return TRC, basins, edges#, lines


    def create_edges(self, TRC):
        """
        Creates edges between different basins and TabulatedRatingCurves (TRC) to build a Ribasim network.

        Parameters:
        - TRC (GeoDataFrame): A GeoDataFrame containing TabulatedRatingCurves information.

        Returns:
        - new_edges (GeoDataFrame): A GeoDataFrame containing the newly created edges with attributes.

        This function takes a GeoDataFrame of TabulatedRatingCurves (TRC) and generates new edges connecting
        basins to TRCs and TRCs back to basins in a Ribasim network. Each TRC edge is split into two, resulting
        in doubled edge count. The new_edges GeoDataFrame is populated with attributes for each edge, including
        'from_node_id', 'to_node_id', 'node_id', and 'geometry'.

        Note:
        The TRC GeoDataFrame is expected to have columns 'from_basin', 'node_id', 'to_basin', 'from_coord', and 'geometry'.
        """
            
        new_edges = gpd.GeoDataFrame(columns=['from_node_id', 'to_node_id', 'node_id', 'geometry'])
        new_edges.node_id = np.arange(1, len(TRC)*2+1) #since the TRC split each edge in half, the total number of new edges should be doubled. +1 since it starts counting at 1


        for i in range(len(TRC)):

            #from basin to TRC
            row = i*2
            new_edges.loc[row, 'from_node_id'] = TRC.from_basin.iloc[i]
            new_edges.loc[row, 'to_node_id'] = TRC.node_id.iloc[i] + len(TRC)

            #add geometry of the new edge
            coord1 = TRC.from_coord.iloc[i]
            coord2 = TRC.geometry.iloc[i]

            new_edges.loc[row, 'geometry'] = LineString([coord1, coord2])


            #from TRC to basin
            row += 1 
            new_edges.loc[row, 'from_node_id'] = TRC.node_id.iloc[i] + len(TRC)
            new_edges.loc[row, 'to_node_id'] = TRC.to_basin.iloc[i]

            #add geometry of the new edge
            coord3 = TRC.geometry.iloc[i]
            coord4 = TRC.to_coord.iloc[i]
            new_edges.loc[row, 'geometry'] = LineString([coord3, coord4])

        new_edges["edge_type"] =  len(new_edges) * ["flow"]
        
        return new_edges

    # Function to create Point objects
    def create_point(self, row):
        coords = row['name'].split('_')
        x = float(coords[0])
        y = float(coords[1])
        return Point(x, y)
    
    
    def read_boundary_data(self, BC_file_path, starttime, unit = 's'):
        """
        Read boundary conditions data from a text file and create a DataFrame.

        This function reads data from a text file that contains boundary conditions
        and creates a DataFrame with columns representing different attributes of the 
        boundary data. The data can be both stationary as well as non-stationary.

        Depending on the timesteps and number of laterals, this can take up several 
        minutes.

        Parameters:
        BC_file_path (str): The path to the text file containing boundary conditions data from D-Hydro.
        boundary_type (str): The boundary types of D-HYDRO. Support currently only 'waterlevelbnd' and 'lateral_discharge'
        Returns:
        pd.DataFrame: A DataFrame containing boundary conditions data organized in columns.
        """

        # Read the data from the file
        with open(BC_file_path, "r") as file:
            lines = [line.strip() for line in file if not line.strip().startswith('#')] #skip lines starting with '#'
        data_str = '\n'.join(lines) # Split the remaining data into individual lines

        lines = data_str.strip().split('\n') # Split the data into individual lines

        boundary_df = pd.DataFrame(columns=['name', 
                                            'function', 
                                            'timeInterpolation', 
                                            'quantity',
                                            'unit',
                                            'static_value'])

        boundary_df.loc[0] = [np.nan] * len(boundary_df.columns) #add first row of nans

        float_pattern = r'^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s+[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$'
        total_lines = len(lines)
        ten_percent_increment = total_lines // 10  # Calculate 10% increment for notification purposes
        count = 0
        dynamic = False
        
        # Loop through each line in the file
        for i in range(len(lines)):
            line = lines[i]

            # Calculate the progress percentage
            if (i + 1) % ten_percent_increment == 0:
                progress_percentage = (i + 1) / total_lines * 100
                print(f"Reading the {progress_percentage:.0f}% of the lines")


            if len(line.replace(" ", "")) == 0 or ('General' or 'fileVersion' or 'fileType') in line: #skip blanc lines
                continue
            elif 'forcing' not in line.replace(" ", "").strip().lower():
                stripped = [item.strip() for item in line.split('=')]
                if len(stripped)==2: #find the key and value, if there is a '=' sign
                    key, value = stripped
                    boundary_df.loc[count, key] = value
                elif boundary_df.iloc[-1]['function'] == 'constant' and len(stripped)==1 and not re.match(float_pattern, line): #check if there is only 1 value, which is not a time series
                    value = line.replace(" ", "")
                    boundary_df.loc[count, 'static_value'] = float(value)
                elif re.match(float_pattern, line) and boundary_df.iloc[-1]['function'] == 'timeseries' or pd.isna(boundary_df.iloc[-1]['function']): #add the time series to the df
                    # Process lines with two floats
                    time_stamp, time_value = [float(item) for item in line.split()]
                    boundary_df.loc[count, 'time_stamp'] = time_stamp
                    boundary_df.loc[count, 'dynamic_value'] = time_value

                    count +=1 #since its a time serie, go to the next row
                    dynamic = True

            else:
                count +=1

        print('All lines are read')
        print()
        #fill a substantial part of the NaN values, and filter out irrelevant rows and columns
        columns_to_fill = ['name', 'function', 'timeInterpolation', 'quantity', 'unit']     #fill columns with the last value for easier look up    # Fill NaN values in specified columns with the previous non-NaN value
        boundary_df[columns_to_fill] = boundary_df[columns_to_fill].fillna(method='ffill')
        boundary_df = boundary_df.loc[~boundary_df.name.isna()]
        
        if dynamic == False:
            boundary_df = boundary_df[['name', 'function', 'timeInterpolation', 'quantity', 'unit', 'static_value']]
        else:
            boundary_df = boundary_df[['name', 'function', 'timeInterpolation', 'quantity', 'unit', 'static_value', 'time_stamp', 'dynamic_value']]
            
            #Convert starttime to datetime, and add the start time to the time columns
            starttime = pd.to_datetime(starttime)
            boundary_df['time_stamp'] = starttime + pd.to_timedelta(boundary_df['time_stamp'], unit='s') * 60 #to do: incorporate more elegant way of defining that the timesteps are in minute
        
        # return boundary_df_static, boundary_df_dynamic
        return boundary_df
    
    def embed_laterals(self, his_file_path, nodes, edges, FlowBoundary, offset):
        """
        Embed lateral flow boundary data into the nodes and edges of a hydraulic model.

        This function takes lateral flow boundary coordinates, a nodes DataFrame, an edges DataFrame,
        and an offset value, and updates the nodes and edges DataFrames to include lateral data
        in the hydraulic model. The offset value is required, so that basins and laterals do not
        overlap. Positive values will be placed north of the basin, and negatives values south.

        Parameters:
        x_coordinates_laterals (list): List of x-coordinates for lateral flow boundaries.
        y_coordinates_laterals (list): List of y-coordinates for lateral flow boundaries.
        nodes (pd.DataFrame): DataFrame containing node information.
        edges (pd.DataFrame): DataFrame containing edge information.
        offset (float): Offset value for lateral flow boundary positions. 

        Returns:
        pd.DataFrame, pd.DataFrame: Updated nodes and edges DataFrames with lateral flow boundary information.
        """
        his_file = xr.open_dataset(his_file_path)
        x_coordinates_laterals = his_file.lateral_geom_node_coordx
        y_coordinates_laterals = his_file.lateral_geom_node_coordy
        
        #To do: assign the x and y coordinates more robust by using the lateral_id within his_file.lateral_geom_node_count
        
        #put the laterals in the correct format
        laterals = pd.DataFrame()
        laterals = gpd.GeoDataFrame(laterals, geometry = gpd.points_from_xy(x_coordinates_laterals, y_coordinates_laterals+offset)) 
        laterals['type'] = 'FlowBoundary'
        laterals['node_id'] = np.arange(max(nodes.node_id)+1, len(laterals)+1+max(nodes.node_id))
        laterals['NAME_ID'] = ['lateral_{}'.format(i) for i in range(1, len(laterals) + 1)]

        #add the laterals to the nodes
        final_nodes = pd.concat([nodes, laterals])
        final_nodes.reset_index(inplace=True, drop=True)
        final_nodes.index = np.arange(1, len(final_nodes)+1)



        #edges
        laterals_to_basin = gpd.GeoDataFrame(laterals, geometry = gpd.points_from_xy(x_coordinates_laterals, y_coordinates_laterals)) #redo this to exclude the offset 
        laterals_to_basin_nodes_ids = pd.merge(nodes, laterals_to_basin, on='geometry').node_id_x
        laterals_to_basin_nodes_ids = laterals_to_basin.merge(right = final_nodes,
                                                     left_on = 'geometry',
                                                     right_on = 'geometry').node_id_y


        edges_laterals = gpd.GeoDataFrame(columns = edges.columns)
        edges_laterals['from_node_id'] = laterals['node_id']
        edges_laterals['from_node_id_new'] = laterals.node_id
        edges_laterals['to_node_id_new'] = laterals_to_basin_nodes_ids
        edges_laterals['to_node_id'] = edges_laterals.to_node_id_new #['lateral_{}'.format(i) for i in range(1, len(final_nodes[final_nodes['type'] == 'FlowBoundary']) + 1)]



        lines = []
        for i in range(len(laterals)):
            a = laterals.geometry.iloc[i].coords[0]


            lateral_point = laterals.geometry.iloc[i].coords[0] 
            edges_lateral_point = laterals_to_basin.geometry.iloc[i].coords[0]
            line = LineString([lateral_point, edges_lateral_point])
            lines.append(line)

        edges_laterals['geometry'] = lines
        edges_laterals['node_id'] = np.arange(max(edges.node_id)+1, len(edges_laterals)+1+max(edges.node_id))

        edges = pd.concat([edges, edges_laterals])   
        final_nodes.drop_duplicates(inplace=True, subset = ['NAME_ID'])
        edges.drop_duplicates(inplace=True, subset = ['from_node_id', 'to_node_id'])
        edges.drop(columns=['from_node_id_new', 'to_node_id_new'], inplace = True)
        # edges.set_index('node_id', inplace=True)
        edges.edge_type = 'flow'
        
        # link it with the FlowBoundary df
        flow_boundary_nodes = final_nodes[final_nodes['NAME_ID'].astype(str).str.startswith('lateral')][['geometry', 'node_id']]
        
        #first take the unique FlowBoundary rows, as the names are duplicate if its a non stationairy simulation. Add it back together.
        FlowBoundary_temp = FlowBoundary.drop_duplicates(subset='name')
        FlowBoundary_temp = FlowBoundary_temp.loc[FlowBoundary_temp['quantity'] == 'lateral_discharge']
        FlowBoundary_temp['geometry'] = flow_boundary_nodes.geometry.values
        FlowBoundary_temp['node_id'] = flow_boundary_nodes.node_id.values
        FlowBoundary_temp = FlowBoundary_temp[['name', 'geometry', 'node_id']]
        # print(FlowBoundary_temp)
        #Merge the temporary df in the original df, and use forward filling to fill the NaN places
        FlowBoundary = FlowBoundary.merge(FlowBoundary_temp, how = 'inner', left_on = 'name', right_on = 'name')
        FlowBoundary = FlowBoundary.fillna('ffill') #forward fill
        FlowBoundary[FlowBoundary == 'ffill'] = np.nan #fill with nans where forward fill is not possible
        # FlowBoundary.set_index('node_id', inplace = True)
        # print(flow_boundary_nodes)
        return final_nodes, edges, FlowBoundary
    
#         def embed_laterals(self, his_file_path, nodes, edges, FlowBoundary, offset):
#         """
#         Embed lateral flow boundary data into the nodes and edges of a hydraulic model.

#         This function takes lateral flow boundary coordinates, a nodes DataFrame, an edges DataFrame,
#         and an offset value, and updates the nodes and edges DataFrames to include lateral data
#         in the hydraulic model. The offset value is required, so that basins and laterals do not
#         overlap. Positive values will be placed north of the basin, and negatives values south.

#         Parameters:
#         x_coordinates_laterals (list): List of x-coordinates for lateral flow boundaries.
#         y_coordinates_laterals (list): List of y-coordinates for lateral flow boundaries.
#         nodes (pd.DataFrame): DataFrame containing node information.
#         edges (pd.DataFrame): DataFrame containing edge information.
#         offset (float): Offset value for lateral flow boundary positions. 

#         Returns:
#         pd.DataFrame, pd.DataFrame: Updated nodes and edges DataFrames with lateral flow boundary information.
#         """
#         his_file = xr.open_dataset(his_file_path)
#         x_coordinates_laterals = his_file.lateral_geom_node_coordx
#         y_coordinates_laterals = his_file.lateral_geom_node_coordy
        
#         #put the laterals in the correct format
#         laterals = pd.DataFrame()
#         laterals = gpd.GeoDataFrame(laterals, geometry = gpd.points_from_xy(x_coordinates_laterals, y_coordinates_laterals+offset)) 
#         laterals['type'] = 'FlowBoundary'
#         laterals['node_id'] = np.arange(max(nodes.node_id)+1, len(laterals)+1+max(nodes.node_id))
#         laterals['NAME_ID'] = ['lateral_{}'.format(i) for i in range(1, len(laterals) + 1)]

#         #add the laterals to the nodes
#         final_nodes = pd.concat([nodes, laterals])
#         final_nodes.reset_index(inplace=True, drop=True)
#         final_nodes.index = np.arange(1, len(final_nodes)+1)



#         #edges
#         laterals_to_basin = gpd.GeoDataFrame(laterals, geometry = gpd.points_from_xy(x_coordinates_laterals, y_coordinates_laterals)) #redo this to exclude the offset 
#         laterals_to_basin_nodes_ids = pd.merge(nodes, laterals_to_basin, on='geometry').node_id_x
#         laterals_to_basin_nodes_ids = laterals_to_basin.merge(right = final_nodes,
#                                                      left_on = 'geometry',
#                                                      right_on = 'geometry').node_id_y


#         edges_laterals = gpd.GeoDataFrame(columns = edges.columns)
#         edges_laterals['from_node_id'] = laterals['node_id']
#         edges_laterals['from_node_id_new'] = laterals.node_id
#         edges_laterals['to_node_id_new'] = laterals_to_basin_nodes_ids
#         edges_laterals['to_node_id'] = edges_laterals.to_node_id_new #['lateral_{}'.format(i) for i in range(1, len(final_nodes[final_nodes['type'] == 'FlowBoundary']) + 1)]



#         lines = []
#         for i in range(len(laterals)):
#             a = laterals.geometry.iloc[i].coords[0]


#             lateral_point = laterals.geometry.iloc[i].coords[0] 
#             edges_lateral_point = laterals_to_basin.geometry.iloc[i].coords[0]
#             line = LineString([lateral_point, edges_lateral_point])
#             lines.append(line)

#         edges_laterals['geometry'] = lines
#         edges_laterals['node_id'] = np.arange(max(edges.node_id)+1, len(edges_laterals)+1+max(edges.node_id))

#         edges = pd.concat([edges, edges_laterals])   
#         final_nodes.drop_duplicates(inplace=True, subset = ['NAME_ID'])
#         edges.drop_duplicates(inplace=True, subset = ['from_node_id', 'to_node_id'])
#         edges.drop(columns=['from_node_id_new', 'to_node_id_new'], inplace = True)
#         # edges.set_index('node_id', inplace=True)
#         edges.edge_type = 'flow'
        
#         # link it with the FlowBoundary df
#         flow_boundary_nodes = final_nodes[final_nodes['NAME_ID'].astype(str).str.startswith('lateral')][['geometry', 'node_id']]
#         FlowBoundary['geometry'] = flow_boundary_nodes.geometry.values
#         FlowBoundary['node_id'] = flow_boundary_nodes.node_id.values
#         # FlowBoundary.set_index('node_id', inplace = True)
#         # print(flow_boundary_nodes)
#         return final_nodes, edges, FlowBoundary
    
    def group_boundary_conditions(self, boundary_df):
        """
        Group boundary conditions from a given DataFrame into static and non-stationary types.

        Parameters:
        - boundary_df (pd.DataFrame): A DataFrame containing boundary condition data.

        Returns:
        - flow_boundary_df_static (pd.DataFrame): Static flow boundary conditions.
        - flow_boundary_df_non_stationary (pd.DataFrame): Non-stationary flow boundary conditions.
        - level_boundary_df_static (pd.DataFrame): Static water level boundary conditions.
        - level_boundary_df_non_stationary (pd.DataFrame): Non-stationary water level boundary conditions.

        This function filters the input DataFrame based on the 'quantity' and 'function'
        columns to separate boundary conditions into different categories.
        """
        #filter on the given type of boundary.           
        flow_boundary_df = boundary_df.loc[boundary_df['quantity'] == 'lateral_discharge']
        flow_boundary_df_static = flow_boundary_df.loc[flow_boundary_df['function'] == 'constant']
        flow_boundary_df_dynamic = flow_boundary_df.loc[flow_boundary_df['function'] == 'timeseries']
        
        level_boundary_df = boundary_df.loc[boundary_df['quantity'] == 'waterlevelbnd']
        level_boundary_df_static = level_boundary_df.loc[level_boundary_df['function'] == 'constant']
        level_boundary_df_dynamic = level_boundary_df.loc[level_boundary_df['function'] == 'timeseries']
        
        return flow_boundary_df_static, flow_boundary_df_dynamic, level_boundary_df_static, level_boundary_df_dynamic
    
    
    def find_Terminal(self, nodes):
        unique_first = np.unique(self.edge_nodes[:, 0]) # Get the unique integers from the first column
        unique_second = np.unique(self.edge_nodes[:, 1]) # Get the unique integers from the second column
        Terminal_basins = np.setdiff1d(unique_second, unique_first) #= the basins which are assumed to be Terminals

        x_coordinates_nodes = self.map_file['mesh1d_s1'].mesh1d_node_x
        y_coordinates_nodes = self.map_file['mesh1d_s1'].mesh1d_node_y
        
        Terminal_df = pd.DataFrame(columns = ['NAME_ID', 'type', 'x_coor', 'y_coor', 'node_id'])
        Terminal_df['NAME_ID'] = ['Terminal_{}'.format(i) for i in range(len(Terminal_basins))] #naam geven
        
        Terminal_df.node_id= Terminal_basins
        Terminal_df.x_coor = x_coordinates_nodes[Terminal_basins-1].values
        Terminal_df.y_coor = y_coordinates_nodes[Terminal_basins-1].values
        Terminal_df.type = 'Terminal'
        Terminal_gdf = gpd.GeoDataFrame(Terminal_df[['NAME_ID', 'type', 'node_id']], geometry = gpd.points_from_xy(Terminal_df.x_coor, Terminal_df.y_coor))

        if len(Terminal_gdf) > 1:
            warnings.warn('There are multiple Terminals in the model. The model generator is not yet tested on this. This may cause problems later in the code. Check whether the orientation of the edges are correct for the following points:')
            print(Terminal_gdf)
        
        Terminal_node_ids = Terminal_gdf['node_id'].values
        nodes.loc[nodes['node_id'].isin(Terminal_node_ids), 'type'] = 'Terminal'
        nodes['NAME_ID'] = nodes['NAME_ID'].str.replace(r'^basin_(\d{4})$', r'Terminal_\1', regex=True)
        # nodes = nodes.set_index('node_id')
        
        return nodes, Terminal_gdf
