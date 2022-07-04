import sys

import loader
import pandas as pd


def load_all_electrode_locations(path):
    # TODO: Make khxx selection more robust
    electrode_filenames = loader.get_filenames(path, 'csv', keywords=['electrode_locations'])

    data = {}
    for file in electrode_filenames:
        # if ppt != None and ppt in str(file):    
        df = pd.read_csv(file)
        data[file.parent.name] = dict(zip(df['electrode_name_1'],
                                       df['location']))
        data[file.parent.name]['order'] = df['electrode_name_1'].to_list()
    return data

def list_locations(files, to_file=sys.stdout):
    ''' to_file: FileObject '''
    
    for ppt, locs in files.items():
        # print('{:<5}: {:<5} contacts | {:<5} Electrodes'\
        #       .format(ppt, len(locs.keys()), -1))
        print(f'{ppt:<4}: {len(locs.keys()):<3}', file=to_file)

if __name__=='__main__':
    path = r'L:\FHML_MHeNs\sEEG'
    files = load_all_electrode_locations(path)
    list_locations(files)