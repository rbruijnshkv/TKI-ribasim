#This file contains all relevant functions for the R-HyDAMO generator


#import relevant libraries
import ribasim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import geopandas as gpd
import xarray as xr
import ugrid as ug
from ugrid import UGrid, UGridNetwork1D
import xugrid as xu
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, LineString
import timeit



def plot_Q_h(Q, h, edges, edge_n, threeD = False):
    """
    Plot the Q of a specified edge, and the corresponding h of the nodes  
    
    Parameters
    ----------
    Q : discharge per node (which is likely to be map_file['mesh1d_q1'])
    h : water level per node (which is likely to be map_file['mesh1d_s1'])
    edges : mesh1d_edge_nodes
    edge_n : edge number to be looked into. Can be any edge you'd like to inspect.
    
    Returns
    ----------
    Plots
    """
   
    start_node = np.array(edges[edge_n])[0] - 1 #minus one since it starts counting at 1 instead of 0
    end_node = np.array(edges[edge_n])[1] - 1 #minus one since it starts counting at 1 instead of 0  
                        
    q1 = Q[:, edge_n] #dicharge at the defined edge_n
    h_up = h[:, start_node]
    h_down = h[:, end_node]    
    
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    
    #upper left, discharge
    q1.plot(ax = axs[0,0], label = 'Discharge [m3/s]')
    axs[0,0].legend()
    axs[0,0].grid()
    axs[0,0].set_title('Discharge')
    
    
    #upper right, waterlevels
    h_up.plot(ax = axs[0,1], label='Waterdepth_upstream [mNAP]')
    h_down.plot(ax = axs[0,1], label='Waterdepth_downstream [mNAP]')
    axs[0,1].legend()
    axs[0,1].grid()
    axs[0,1].set_title('Water levels')
    
    
    #middle left, waterlevel upstream
    scatter = axs[1,0].scatter(x = h_up,
                               y = q1,
                               c =plt.cm.viridis(np.linspace(0, 1, len(h_up))),
                               alpha = 0.5)
    axs[1,0].set_xlabel('Waterdepth_upstream [mNAP]')
    axs[1,0].set_ylabel('Discharge [m3/s]')
    axs[1,0].grid()
    axs[1,0].set_title('Q(h) relation with upstream node')
    fig.colorbar(scatter, ax=axs[1,0], label = 'Relative time stamp')

    
    #middle right, waterlevel downstream
    scatter = axs[1,1].scatter(x = h_down,
                               y = q1,
                               c =plt.cm.viridis(np.linspace(0, 1, len(h_down))),
                               alpha = 0.5)
    axs[1,1].set_xlabel('Waterdepth_upstream [mNAP]')
    axs[1,1].set_ylabel('Discharge [m3/s]')
    axs[1,1].grid()
    axs[1,1].set_title('Q(h) relation with downstream node')
    fig.colorbar(scatter, ax=axs[1,1], label = 'Relative time stamp')


    #lower left, Q(h)-relation downstream
    scatter = axs[2,0].scatter(x = h_down,
                               y = h_up,
                               c = q1,
                               alpha = 0.2)
    axs[2,0].set_xlabel('Waterdepth_upstream [mNAP]')
    axs[2,0].set_ylabel('Waterdepth_downstream [mNAP]')
    axs[2,0].grid()
    axs[2,0].set_title('Q(h) relation')
    fig.colorbar(scatter, ax=axs[2,0], label = 'Discharge [m3/s]')

    
        
    if threeD == True:
        size = 0.01
        axs[2,1].set_axis_off()
        
        ax = fig.add_subplot(3, 2, 6, projection='3d')
        colors = q1 / np.max(q1)  # normalize q1 values to [0, 1] range

        ax.bar3d(x = np.array(h_up),
                       y = np.array(h_down),
                       z = np.zeros_like(q1),
                       dx = np.ones_like(h_up)*size,
                       dy = np.ones_like(h_up)*size,
                       dz = np.array(q1),
                       color=plt.cm.viridis(colors),
                       alpha = 0.2)
        # ax.view_init(elev=30, azim=45)
        # Set the axis labels
        ax.set_xlabel('Waterdepth_upstream [mNAP]')
        ax.set_ylabel('Waterdepth_downstream [mNAP]')
        ax.set_zlabel('Discharge [m3/s]')

        # Adjust the viewing angle
        ax.view_init(roll=0)

        # Add grid lines
        ax.grid(True)
    
    fig.patch.set_facecolor('none')
    plt.tight_layout()
    # plt.show()
    
    
    #check whether the coordinates of the the edges and node correspond
    snxl, snyl = h[:, start_node].mesh1d_node_x.values, h[:, start_node].mesh1d_node_y.values
    enxl, enyl = h[:, end_node].mesh1d_node_x.values, h[:, end_node].mesh1d_node_y.values
    exl, eyl = Q[:, edge_n].mesh1d_edge_x.values, Q[:, edge_n].mesh1d_edge_y.values
    
    x, y = True, True
    
    if exl > np.mean([snxl, enxl])+1 or exl < np.mean([snxl, enxl])-1:
        x = False
    elif eyl > np.mean([snyl, enyl])+1 or eyl < np.mean([snyl, enyl])-1:
        y = False
        
    if x == False or y == False:    
        print('x =', x)
        print('y =', y)
        print('Edge x location: ', int(Q[:, edge_n].mesh1d_edge_x.values))
        print('Edge y location: ', int(Q[:, edge_n].mesh1d_edge_y.values))
        print()
        print('Start node x location: ', int(h[:, start_node].mesh1d_node_x.values))
        print('Start node y location: ', int(h[:, start_node].mesh1d_node_y.values))
        print()
        print('End node x location: ', int(h[:, end_node].mesh1d_node_x.values))
        print('End node y location: ', int(h[:, end_node].mesh1d_node_y.values))
    else:
        print('Coordinates of the nodes and edges are likely to be correct')
    
                           

        
        
#Convert the coordinates of all calculation points to Basin Nodes. #basins, edges, TRC
def attribute_converter(x_coordinates_nodes, 
                        y_coordinates_nodes, 
                        edges):
    
    
    #coordinates of the basin
    basins = pd.DataFrame()
    basins['x_coordinate'], basins['y_coordinate'] = x_coordinates_nodes, y_coordinates_nodes
    #to do: add name of each node here
    basins = gpd.GeoDataFrame(basins, geometry = gpd.points_from_xy(basins['x_coordinate'], basins['y_coordinate']))

    #To do: robuuster maken. Gelijk vanaf het begin naam van nodes en edges toevoegen, en op basis daarvan indexen
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

    return TRC, basins, lines





def BoundaryNode_Terminal(x_coordinates_nodes, y_coordinates_nodes, edges):                                                                #BoundaryNode was eerste LevelControl
    edges = edges.values #n times 2 array
    unique_first = np.unique(edges[:, 0]) # Get the unique integers from the first column
    unique_second = np.unique(edges[:, 1]) # Get the unique integers from the second column
    
    
    #Fill BoundaryNode with coordinates
    BoundaryNode_basins = np.setdiff1d(unique_first, unique_second) #= the basins which are assumed to be BoundaryNode 
    BoundaryNode_df = pd.DataFrame(columns = ['NAME_ID', 'type', 'x_coor', 'y_coor', 'basin_n', 'node_id'])
    BoundaryNode_df['NAME_ID'] = ['BoundaryNode_{}'.format(i) for i in range(len(BoundaryNode_basins))] #naam geven
    
    BoundaryNode_df.basin_n= BoundaryNode_basins
    BoundaryNode_df.x_coor = x_coordinates_nodes[BoundaryNode_basins-1].values
    BoundaryNode_df.y_coor = y_coordinates_nodes[BoundaryNode_basins-1].values
    BoundaryNode_df.type = 'BoundaryNode' 

    BoundaryNode_gdf = gpd.GeoDataFrame(BoundaryNode_df[['NAME_ID', 'type', 'basin_n', 'node_id']], geometry = gpd.points_from_xy(BoundaryNode_df.x_coor, BoundaryNode_df.y_coor))
    BoundaryNode_gdf = BoundaryNode_gdf
    
    #Fill Terminal with coordinates
    Terminal_basins = np.setdiff1d(unique_second, unique_first) #= the basins which are assumed to be Terminals
    Terminal_df = pd.DataFrame(columns = ['NAME_ID', 'type', 'x_coor', 'y_coor', 'basin_n', 'node_id'])
    Terminal_df['NAME_ID'] = ['Terminal_{}'.format(i) for i in range(len(Terminal_basins))] #naam geven
   
    Terminal_df.basin_n= Terminal_basins
    Terminal_df.x_coor = x_coordinates_nodes[Terminal_basins-1].values
    Terminal_df.y_coor = y_coordinates_nodes[Terminal_basins-1].values
    Terminal_df.type = 'Terminal'
    
    Terminal_gdf = gpd.GeoDataFrame(Terminal_df[['NAME_ID', 'type', 'basin_n', 'node_id']], geometry = gpd.points_from_xy(Terminal_df.x_coor, Terminal_df.y_coor))
 
    return BoundaryNode_gdf, Terminal_gdf



def fill_Q_h(Q, h, edges, TRC, upstream = True): #Add TRC for the names
    timesteps = len(Q)
    number_of_TRCs = len(edges)
    
    TRC_table = pd.DataFrame(columns =['discharge', 'level', 'node_id'])
    TRC_table.node_id = np.repeat(np.arange(1, number_of_TRCs + 1), timesteps)
    start_time = timeit.default_timer()
    
    for i in range(number_of_TRCs-1):        
        if upstream == True:
            node = np.array(edges[i])[0] - 1 #minus one since it starts counting at 1 instead of 0. i = node, [0] is the "from" node
        else:
            node = np.array(edges[i])[1] - 1 #minus one since it starts counting at 1 instead of 0. i = node, [0] is the "to" node

        Q_basin, h_basin = Q[:,node], h[:,node] #retrieve Q and h from a specific basin. 

        node += 1 #+1 again to store them in the correct node_id row
        TRC_table.loc[TRC_table.node_id == node, 'discharge'] = Q_basin #fill Q
        TRC_table.loc[TRC_table.node_id == node, 'level'] = h_basin #fill h
        
    TRC_table.node_id += len(Q[0]) + 1 #tel bij elke TRC het aantal basins + +1, net zoals in def attribute_converter    
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print("Execution time filling the entire TRC:", execution_time, "seconds")
    
    TRC_table['discharge'] = TRC_table['discharge'].astype(float)
    TRC_table['level'] = TRC_table['level'].astype(float)
    return TRC_table


#There are duplicate points between LevelControl and basins, which will be removed by following code

def filter_basins_LevelControl(basins, LevelControl):
    # Convert the 'basin_n' column to string in the LevelControl GeoDataFrame
    LevelControl['basin_n'] = LevelControl['basin_n'].astype(str)

    # Extract the basin numbers from the 'basin_n' column
    basin_numbers = LevelControl['basin_n'].tolist()

    # Filter the rows in the basins GeoDataFrame based on the 'NAME_ID' column
    filtered_basins = basins[basins['NAME_ID'].str.extract('(\d+)', expand=False).astype(str).isin(basin_numbers)]

    # Drop the selected rows from the basins GeoDataFrame
    filtered_basins = basins.drop(filtered_basins.index)
    
    return filtered_basins




def filter_TRC_Terminal(TRC_table, Terminal):
    rows_to_drop = TRC_table[TRC_table['node_id'].astype(str).isin(Terminal['basin_n'])].index
    filtered_TRC_table = TRC_table.drop(rows_to_drop)
    return filtered_TRC_table




def updating_edges(edges, TRC):
    new_edges = gpd.GeoDataFrame(columns=['from_node_id', 'to_node_id', 'fid', 'geometry'])
    new_edges.fid = np.arange(1, len(TRC)*2+1) #since the TRC split each edge in half, the total number of new edges should be doubled. +1 since it starts counting at 1
    
    
    for i in range(len(TRC)):
    # for i in range(1000):
        #make a distinction between iterating (i) through TRC, and iterating (per row) through the new_edges
        #from basin to TRC
        row = i*2
        new_edges.loc[row, 'from_node_id'] = TRC.from_node.iloc[i]
        new_edges.loc[row, 'to_node_id'] = TRC.NAME_ID.iloc[i]
        
        #add geometry of the new edge
        x1, y1 = TRC.x_s_node.iloc[i], TRC.y_s_node.iloc[i]
        x2, y2 = TRC.x_TRC.iloc[i], TRC.y_TRC.iloc[i]
        new_edges.loc[row, 'geometry'] = LineString([(x1, y1), (x2, y2)])
        
        # new_edges['geometry'] = TRC.apply(lambda row: LineString([(row['x_s_node'], row['y_s_node']), (row['x_e_node'], row['y_e_node'])]), axis=1)

        
        
        #from TRC to basin
        row += 1 
        new_edges.loc[row, 'from_node_id'] = TRC.NAME_ID.iloc[i]
        new_edges.loc[row, 'to_node_id'] = TRC.to_node.iloc[i]
              
        #add geometry of the new edge
        x1, y1 = x2, y2
        x2, y2 = TRC.x_e_node.iloc[i], TRC.y_e_node.iloc[i]
        new_edges.loc[row, 'geometry'] = LineString([(x1, y1), (x2, y2)])
                                                    
    return new_edges