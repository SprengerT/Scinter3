import numpy as np
import os
import sys

sys.path.insert(0,'/path/to/Scinter3')
import scinter_import
#import scinter_data
#import scinter_computation
#import scinter_plot
import scinter_measurement

path_data = "/path/to/your/data"
path_results = "/path/chosen/by/you/for/processed/data/and/results"

obsname = "label_your_observation"
filename = os.path.join(path_data,"name_of_your_data_file.npz")

"""
Add this class to scinter_import:

class from_npz(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        data_npz = np.load(data_path)
        self.DS = data_npz["DS"]
        self.nu = data_npz["nu"]
        self.mjd = data_npz["mjd"]
        self.DS = self.DS/np.std(self.DS)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
"""

DS = scinter_import.from_npz(filename)

"""
The next step is to save the data for easy access and an organized structure of the results.
This just creates another copy in this case, but the idea is that you change the format
of the data and/or crop, downsample, or rescale the data here for later use.
"""

# This path will contain all observations of this group, e.g. the same source.
storage = scinter_measurement.storage(path_results)
# This step creates the directory for the new observation.
obs = storage.add_obs(obsname)
# This step creates the data and info files for the new observation.
obs.log_DS(DS)
# Optional step to add this observation to a subgroup list.
# One observation can belong to multiple lists and by default is already added to the 'all' list.
storage.addtolist("your_list",obsname)

"""
Basic information about the observation can be found in computations/logfile.yaml
"""