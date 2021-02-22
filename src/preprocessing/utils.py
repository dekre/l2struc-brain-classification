import numpy as np
import zipfile as zip
import pandas as pd
from pydicom import dcmread



def get_labels(lables_path: str) -> pd.DataFrame:
    data = pd.read_csv(lables_path, sep = ";")
    data["y"] = 0
    data.loc[data.group == "L2", "y"] = 1
    data.loc[data.group.isnull(), "y"] = np.NaN
    return data

def get_data(srczip: zip.ZipFile, lables_path: str) -> pd.DataFrame:    
    data = get_labels(lables_path)
    tmp = list()
    for file in srczip.namelist():
        if file.endswith(".dcm"):
            subject = file.split("/")[1]
            tmp.append((int(subject), file))
    data = (
        data
        .merge(
            pd.DataFrame(tmp, columns = ['subject', 'file']),
            on = "subject"            
        )
    )
    return data[data.subject != '105']
            
