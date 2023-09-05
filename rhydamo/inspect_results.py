class inspect_results():
    def __init__(self):
        pass
    
    def post_process_h_results(self, basin_overview, h_DHYDRO, h_Ribasim):
        """
        Post-processes water level data from D-HYDRO and Ribasim simulations.

        This function takes water level data from D-HYDRO and Ribasim simulations,
        along with basin overview information, and performs necessary post-processing
        to format and organize the data for comparison.

        Parameters:
            basin_overview (geopandas.GeoDataFrame): A GeoDataFrame containing basin overview
                information including node IDs and geometries.
            h_DHYDRO (xarray.Dataset): Water level data from D-HYDRO simulation.
            h_Ribasim (pd.DataFrame): Water level data from Ribasim simulation.

        Returns:
            tuple: A tuple containing two dataframes:
                - h_Ribasim (pd.DataFrame): Post-processed Ribasim water level data.
                - h_DHYDRO (pd.DataFrame): Post-processed D-HYDRO water level data.
        """

        h_coords = gpd.GeoDataFrame(geometry = gpd.points_from_xy(results.mesh1d_s1.coords['mesh1d_node_x'], results.mesh1d_s1.coords['mesh1d_node_y']))
        find_basins = h_coords.merge(basin_overview, on='geometry')[['geometry', 'node_id']]

        #post process the data format of the water level of D-HYDRO
        h_DHYDRO = results.mesh1d_s1
        h_DHYDRO = h_DHYDRO.to_dataframe()
        h_DHYDRO.sort_values(by=['mesh1d_nNodes', 'time'], inplace = True)
        geometry = gpd.GeoDataFrame(geometry = gpd.points_from_xy(h_DHYDRO['mesh1d_node_x'].iloc[:], h_DHYDRO['mesh1d_node_y'].iloc[:]))

        geometry.set_index(h_DHYDRO.index, inplace=True) # Set the index of the geometry GeoDataFrame to match the MultiIndex of h_DHYDRO
        h_DHYDRO['geometry'] = geometry

        #rename and filter columns        
        h_DHYDRO.rename(columns={'mesh1d_s1': 'level_DHYDRO'}, inplace=True) #change names for better comparison
        h_DHYDRO.index.set_names({'mesh1d_nNodes': 'node_id'}, inplace = True)
        h_DHYDRO = h_DHYDRO[['level_DHYDRO', 'geometry']]
        h_DHYDRO = h_DHYDRO.reset_index(level=[0,1])
        h_DHYDRO.node_id += 1

        #post process the data format of the water level of Ribasim
        if 'node_id' in h_Ribasim.columns:
            h_Ribasim.drop('node_id', axis=1, inplace=True)
        h_Ribasim = h_Ribasim.reset_index(level=[0,1])
        h_Ribasim = h_Ribasim.merge(basin_overview, left_on =h_Ribasim['node_id'], right_on='node_id')[['node_id', 'time', 'storage', 'level', 'geometry']]
        h_Ribasim.rename(columns={'level': 'level_Ribasim'}, inplace = True)
        h_Ribasim.sort_values(by=['node_id', 'time'])

        return h_Ribasim, h_DHYDRO
