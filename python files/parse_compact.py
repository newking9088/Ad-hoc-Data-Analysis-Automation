import bz2 
import json 
import pandas as pd 

def parse_bz2(file):
    with bz2.open(file) as f:
        dftemp=[json.loads(line) for line in f]
    df=pd.DataFrame(dftemp)
    data = pd.DataFrame.from_records(df.data.values)
    df = df.drop('data', axis=1) 
    df = pd.concat([df, data], axis=1)
  
    #map msg_type numbers
    msg_dict = {
            14: 'ERROR',
            54: 'CHARGE',
            37: 'CHARGE',
            38: 'CHARGE',
            40: 'DCIR',
            41: 'DCIR_VALUE',
            42: 'DISCHARGE',
            43: 'REST',
            46: 'CHARGE',
            35: 'CHARGE',
            24: 'STATUS',
            59: 'PACK_DATA',
            50: 'WAVEFORM_CURRENT',
            51: 'WAVEFORM_VOLTAGE',
            61: 'OLIP_STATE',
            63: 'ICA'
        }
    for msg in [x for x in df.msg_type.unique() if x not in msg_dict.keys()]:
        msg_dict[msg] = msg 

    df['msg_type'] = df.msg_type.map(msg_dict)
    df = df.sort_values('time_stamp') 
    df = df.reset_index(drop=True)

    return df
