#This file contains all relevant functions for the R-HyDAMO generator

def plot_Q_h(Q, h, edges, edge_n, threeD = False):
    """
    Plot the Q of a specified edge, and the corresponding h of the nodes  
    
    Parameters
    ----------
    Q : mesh1d_q1
    h : mesh1d_s1
    edges : mesh1d_edge_nodes
    edge_n : edge number to be looked into
    
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
    
                           
                    