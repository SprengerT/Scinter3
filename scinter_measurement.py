import numpy as np
from numpy import newaxis as na
import os
from ruamel.yaml import YAML

import scinter_data

class storage:
    def __init__(self,data_path):
        """
        organizing the a collection of observations
        """
        self.data_path = data_path
        self.yaml = YAML(typ='safe')
        self.lists_file = os.path.join(self.data_path,"obs_lists.yaml")
        if os.path.exists(self.lists_file):
            with open(self.lists_file,'r') as readfile:
                self.obs_lists = self.yaml.load(readfile)
        else:
            self.obs_lists = {"all":[]}
            with open(self.lists_file,'w') as writefile:
                self.yaml.dump(self.obs_lists,writefile)
        
    def add_obs(self,obsname):
        obs = measurement(self.data_path,obsname)
        self.addtolist("all",obsname)
        return obs
        
    def addtolist(self,listname,obsname):
        if listname in self.obs_lists:
            new_list = self.obs_lists[listname]
            if not obsname in new_list:
                new_list.append(obsname)
            try:
                # try to sort the list by date
                mjds = self.get_array(listname,"mjd0")
                new_list = [x for _, x in sorted(zip(mjds, new_list))]
            except:
                pass
            self.obs_lists.update({listname:new_list})
        else:
            self.obs_lists.update({listname:[obsname]})
        with open(self.lists_file,'w') as writefile:
            self.yaml.dump(self.obs_lists,writefile)
            
    def get_array(self,listname,key):
        obs_list = self.obs_lists[listname]
        arr = []
        for obsname in obs_list:
            obs = measurement(self.data_path,obsname)
            arr.append(obs.results[key])
        return np.array(arr)

class measurement:
    def __init__(self,obs_path,name):
        """
        lightweight object to organize file structures and deduced results
        self.name: name to identify the observation
        self.data_path: parent directory of all results and data_transformation
        self.logfile: yaml text file containing single number results
        self.results: dictionary of single number results and identifiers
        """
        self.name = name
        self.obs_path = os.path.join(obs_path,name)
        self.data_path = os.path.join(self.obs_path,"computations")
        self.yaml = YAML(typ='safe')
        self.logfile = os.path.join(self.data_path,"logfile.yaml")
        if not os.path.exists(self.obs_path):
            os.mkdir(self.obs_path)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if os.path.exists(self.logfile):
            with open(self.logfile,'r') as readfile:
                self.results = self.yaml.load(readfile)
        else:
            self.results = {}
            with open(self.logfile,'w') as writefile:
                self.yaml.dump(self.results,writefile)
                
    def log_DS(self,DS):
        fname = os.path.join(self.data_path,"DS.npz")
        np.savez(fname,t=DS.t,mjd=DS.mjd,nu=DS.nu,DS=DS.DS)
        print("New DS saved as "+fname)
        self.results.update({"mjd0":float(DS.mjd0)})
        self.results.update({"nu0":float(DS.nu0)})
        self.results.update({"timespan":float(DS.timespan)})
        self.results.update({"bandwidth":float(DS.bandwidth)})
        with open(self.logfile,'w') as writefile:
            self.yaml.dump(self.results,writefile)
            
    def enter_result(self,key,value,dtype=float):
        self.results.update({key:dtype(value)})
        with open(self.logfile,'w') as writefile:
            self.yaml.dump(self.results,writefile)