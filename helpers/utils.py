import pickle,os
import numpy as nnp

def sigmoid(x: nnp.ndarray) -> nnp.ndarray:
    """Map real-valued logits to probabilities."""
    return 1 / (1 + nnp.exp(-x))

def getIndex(which:str='particle',feat:str=None)->int:
    if which=='particle':
        nameArray=particleFeatureNames
    elif which=='jet':
        nameArray=jetFeatureNames
    else:
        print("arg which must be either particle or jet. Returning -1 as index")
        return -1
    try:
        idx=nameArray.index(feat)
    except ValueError as v:
        print(f"item {feat} not found in list {nameArray}")
        idx=-1
    return idx

def print_events(events,name=None):
    '''Function for printing nested dictionary with at most 3 levels, with final value being a numpy.ndarry, prints the shape of the array'''
    if name: print (name)
    for channel in events:
        if type(events[channel]) == nnp.ndarray or type(events[channel]) == list:
            if type(events[channel]) == nnp.ndarray or channel=='EventAttribute': print ("    Final State:", channel,nnp.array(events[channel]).shape)
            else: 
                try: print ("    Final State:", channel,[item.shape for item in events[channel]])
                except AttributeError:
                    print ("    Final State:", channel,len(events[channel]))
            continue
        print ("Channel: ",channel)
        if type(events[channel]) != dict: continue
        for topology in events[channel]:
            if type(events[channel][topology])!= dict: continue
            if type(events[channel][topology])==nnp.ndarray  or type(events[channel]) == list:
                print ("    Final State: ",topology, nnp.array(events[channel][topology]).shape)
                continue
            print ("Topology: ",topology)
            for final_state in events[channel][topology]:
                print ("    Final State: ",final_state," Shape: ",events[channel][topology][final_state].shape)
    return

def Unpickle(path=None):
    with open(path,'rb') as f:
        return_object=pickle.load(f)
    return return_object

def Pickle(python_object,filename,path=None,save_path=".",verbose=True,overwrite=True,append=False,extension='.pickle'):
    '''save <python_object> to <filename> at location <save_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: save_path=path
    file_path=os.path.join(save_path,filename)
    if not overwrite:
        if os.path.exists(file_path):
            raise IOError("File already exists!")
    if append: 
        assert type(python_object)==dict
        prev=Unpickle(file_path)
        print_events(prev,name="old")
        python_object=merge_flat_dict(prev,python_object)
        print_events(python_object,name="appended")
    if type(python_object)==nnp.ndarray:
        nnp.save(file_path,python_object)
        suffix=".npy"
    else:
        try:
            with open(file_path,"wb") as stream:
                pickle.dump(python_object,stream)
        except OverflowError as e:
            os.remove(file_path)
            print (e,"trying to save as numpy arrays in folder...")
            folder_save(python_object,filename.split(".")[0],save_path)
            return
        suffix=""
    if verbose: print (filename+suffix, " saved at ", os.path.abspath(save_path))
    return

def folder_save(events,folder_name,save_path,append=False):
    pwd=os.getcwd()
    os.chdir(save_path) 
    try: os.mkdir(folder_name)
    except FileExistsError as e: 
        print (e,"Overwriting...")
    finally:os.chdir(folder_name)                      
    for item in events: 
        if append:
            print ("appending...") 
            events[item]=nnp.concatenate((nnp.load(item+".npy",allow_pickle=True),events[item]),axis=0)
        if type(events[item]) ==list:
            print("list type found as val, creating directory...")
            os.mkdir(item)
            os.chdir(item)
            for i,array in enumerate(events[item]):
                nnp.save(item+str(i),array,allow_pickle=True)
                print (array.shape,"saved at ",os.getcwd())
            os.chdir("..")
        else: 
            nnp.save(item,events[item],allow_pickle=True)
            print (item+".npy saved at ",os.getcwd(), "shape = ",events[item].shape)
    os.chdir(pwd)
    return

particleFeatureNames:list[str]=['eta', 'phi', 'pt']

jetFeatureNames=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy',                                                                                    
       'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2',                                                                          
       'jet_tau3', 'jet_tau4']

feature_limits={'eta':{'min':-nnp.pi,'max':nnp.pi},'phi':{'min':-nnp.pi,'max':nnp.pi},'pt':{'min':0,'max':1.0}}
# Sample ranges the readers' fixed_rescale maps onto feature_limits. The eta/phi
# entries are the jet radius: JetClass clusters anti-kt R=0.8 jets, JetsGame R=1.0,
# so each dataset's constituents span a different jet-relative (eta, phi) window
# and needs its own assumed range to land on [-pi, pi).
assumed_limits={'pt':[1.0e-4,3000.],'eta':[-0.8,0.8],'phi':[-0.8,0.8]}
jetsgame_assumed_limits={'pt':[1.0e-4,3000.],'eta':[-1.0,1.0],'phi':[-1.0,1.0]}
