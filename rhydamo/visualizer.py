import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import numpy as np
import random
import os 

class plotter:
    def __init__(self):
        pass
    
    def basins_TRC(basins, TRC):
        fig, ax = plt.subplots()
        TRC.plot(ax=ax, color = 'limegreen', alpha = .5, markersize = 1, label = "TRC's")
        basins.plot(ax=ax, color = 'cornflowerblue', alpha = .5, markersize = 1, label = 'Basins')
        ctx.add_basemap(ax, crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('First impression of the Ribasim schematisation')
        ax.legend()
        
    def plot_basins_TRC_with_zoom(basins, TRC, zoom = 0.025):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create a 1x2 grid of subplots
        fig.suptitle('Ribasim Schematisation')

        # Plot overall view
        axs[0].set_title('Overall View')
        TRC.plot(ax=axs[0], color='limegreen', alpha=.5, markersize=10, label="TRC's")
        basins.plot(ax=axs[0], color='cornflowerblue', alpha=.5, markersize=10, label='Basins')
        ctx.add_basemap(axs[0], crs='EPSG:28992', source=ctx.providers.OpenStreetMap.Mapnik)
        axs[0].legend()

        # Calculate the bounding box for the zoomed-in view (5% of extent)
        xmin, ymin, xmax, ymax = basins.total_bounds
        xmin_new = (xmin+xmax)/2 - zoom * (xmax - xmin)
        xmax_new = (xmin+xmax)/2 + zoom * (xmax - xmin)
                
        ymin_new = (ymin+ymax)/2 - zoom * (ymax - ymin)
        ymax_new = (ymin+ymax)/2 + zoom * (ymax - ymin)

        # Plot zoomed-in view
        axs[1].set_title('Zoomed View')
        TRC.plot(ax=axs[1], color='limegreen', alpha=.5, markersize=10, label="TRC's")
        basins.plot(ax=axs[1], color='cornflowerblue', alpha=.5, markersize=10, label='Basins')
        ctx.add_basemap(axs[1], crs='EPSG:28992', source=ctx.providers.OpenStreetMap.Mapnik)
        axs[1].set_xlim(xmin_new, xmax_new)  # Set x-axis limits
        axs[1].set_ylim(ymin_new, ymax_new)  # Set y-axis limits
        axs[1].legend()
        
    def plot_edges(edges):
        fig, ax = plt.subplots()
        edges.plot(ax=ax, color = 'blue', alpha = .5, markersize = 1, label = "Edges")
        ctx.add_basemap(ax, crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('First impression of the Ribasim edges')
        ax.legend()
        
        
    def VolumeTool_impression(VT):
        
        """
        Plot various parameters from a Volume Tool (VT) dataset.

        Parameters:
        VT (xarray.Dataset): The Volume Tool dataset containing required variables.

        Returns:
        None
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        # Plot 1: Bedlevel
        scatter1 = axes[0, 0].scatter(x=VT.mesh1d_node_x, y=VT.mesh1d_node_y, c=VT.bedlevel, cmap= 'terrain')
        axes[0, 0].set_title('Bedlevel')
        ctx.add_basemap(axes[0, 0], crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik, alpha = 0.35)
        fig.colorbar(scatter1, ax=axes[0, 0])

        # Plot 2: Topheight
        scatter2 = axes[0, 1].scatter(x=VT.mesh1d_node_x, y=VT.mesh1d_node_y, c=VT.topheight, cmap= 'Spectral_r')
        axes[0, 1].set_title('Topheight')
        ctx.add_basemap(axes[0, 1], crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik, alpha = 0.35)
        fig.colorbar(scatter2, ax=axes[0, 1])

        # Plot 3: Max volume
        scatter3 = axes[1, 0].scatter(x=VT.mesh1d_node_x, y=VT.mesh1d_node_y, c=np.max(VT.volume, axis=1), vmax=1000, cmap = 'Blues')
        axes[1, 0].set_title('Max volume')
        ctx.add_basemap(axes[1, 0], crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik, alpha = 0.35)
        fig.colorbar(scatter3, ax=axes[1, 0])

        # Plot 4: Dead storage
        scatter4 = axes[1, 1].scatter(x=VT.mesh1d_node_x, y=VT.mesh1d_node_y, c=np.max(VT.deadstorage, axis=1), cmap='gray_r')
        axes[1, 1].set_title('Dead storage')
        ctx.add_basemap(axes[1, 1], crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik, alpha = 0.35)
        fig.colorbar(scatter4, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()
        
    def profile_sample(profiles, node_id_of_interest=False):
        if node_id_of_interest == False:
            NoI = random.choice(profiles.node_id.unique())
        else:
            NoI = node_id_of_interest
            
        profiles.loc[profiles.node_id == NoI].plot(x='level', y='area', kind='line', marker='o', color = 'gray', figsize=(10, 6), legend=False)
        plt.xlabel('Level')
        plt.ylabel('Area')
        plt.title(f'Level vs. Area for node {NoI}')
        plt.show()

    
    def visualize_changed_Qh(Qh, changed_Qh_index, store_additional_information):
        alterated_Qh_index = changed_Qh_index
        Qh_post_processed = Qh
        
        alterated_Qh = Qh.iloc[alterated_Qh_index]
        Qh_post_processed_plot = gpd.GeoDataFrame(Qh_post_processed, geometry = 'level_coord')
        alterated_Qh = gpd.GeoDataFrame(alterated_Qh, geometry = 'level_coord')

        fig, ax = plt.subplots()
        Qh_post_processed_plot.plot(ax=ax, color = 'cornflowerblue', alpha = .5, markersize = 1, label = 'Actual Qh relations')
        alterated_Qh.plot(ax=ax, color = 'red', alpha = .5, markersize = 1, label = 'Dummy Qh relations')
        ctx.add_basemap(ax, crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('Locations of the adapted Qh relations')
        ax.legend()

        if store_additional_information == True:
            if not os.path.exists('Results\Interim_results'):
               os.makedirs('Results\Interim_results')
            filename = 'Results/Interim_results/' + ax.get_title() + '.png'
            plt.savefig(filename, bbox_inches='tight')

            if not os.path.exists('Results\Interim_data'):
               os.makedirs('Results\Interim_data')

            alterated_Qh[['node_id', 'level_coord']].to_file(r'Results\Interim_data\alterated_Qh_locations.shp')


    def visualize_elevated_Qh(Qh, changed_Qh_height, changed_Qh_index):
        Qh_post_processed = Qh
        Delta_h_low_node_id = changed_Qh_height
        alterated_Qh_index=changed_Qh_index
        
        alterated_Qh = Qh.iloc[alterated_Qh_index]
        Qh_post_processed_plot = gpd.GeoDataFrame(Qh_post_processed, geometry = 'level_coord')
        alterated_Qh = gpd.GeoDataFrame(alterated_Qh, geometry = 'level_coord')

        new_manning_nodes = Qh_post_processed.loc[Qh_post_processed.node_id.isin(Delta_h_low_node_id)]
        new_manning_nodes = gpd.GeoDataFrame(new_manning_nodes, geometry = 'level_coord')
        
        
        fig, ax = plt.subplots()
        Qh_post_processed_plot.plot(ax=ax, color = 'cornflowerblue', alpha = .5, markersize = 1, label = 'Actual Qh relations')
        new_manning_nodes.plot(ax=ax, color = 'orange', alpha = .5, markersize = 1, label = 'Too low dh, changed to Manning')
        ctx.add_basemap(ax, crs='EPSG:28992',source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('Locations of the adapted Qh relations to Manning')
        ax.legend()

        # new_manning_nodes.to_file(r'Results\Interim_data\new_manning_nodes.shp')
        # new_manning_nodes[['node_id', 'level_coord']].to_file(r'Results\Interim_data\new_manning_nodes2.shp')

