import pandas as pd

class forcing_generator:
    def __init__(self):
        pass

    def create_forcing(self, basins, terminal, dummy = True,):
        """
        Create forcing data for drainage calculation based on the specified basins.

        Parameters:
        basins (pandas.DataFrame): The DataFrame containing basin information.
                                   It should at least have a 'node_id' column.
        dummy (bool, optional): Whether to use dummy forcing values. Defaults to True.

        Returns:
        pandas.DataFrame: A DataFrame containing forcing data for each basin, including columns
                          for 'drainage', 'potential_evaporation', 'infiltration', 'precipitation', and 'urban_runoff'.

        """
        #dummy forcing for the drainage calculation
        seconds_in_day = 24 * 3600
        precipitation = 0.000 / seconds_in_day
        evaporation = 0.000 / seconds_in_day

        static = pd.DataFrame(columns= ["node_id", "drainage", "potential_evaporation", "infiltration", "precipitation", "urban_runoff"])
        static.node_id = basins.node_id.unique()
        static = static.fillna(0.0)
        static.index +=1
        
        static = static.loc[static.node_id != terminal.node_id.values[0]] #dont include the terminal
        
        # static.set_index('node_id', drop=True, inplace=True)
        
        return static
