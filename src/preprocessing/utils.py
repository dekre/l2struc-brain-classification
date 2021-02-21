import numpy as np
import zipfile as zip
import pandas as pd
from pydicom import dcmread

def get_brain_slice(pixel_array: np.ndarray, pos: tuple = (0,0), segments: tuple = (8,8)):    
    total_pxiel = np.array(pixel_array.shape)
    pos, segments = np.array(pos), np.array(segments)
    rng_start = (total_pxiel / segments * pos).astype(int)
    rng_end = (rng_start + total_pxiel / segments).astype(int)    
    return pixel_array[rng_start[0]:rng_end[0],rng_start[1]:rng_end[1]]


def get_brain_image_pixel_array(src_zip: zip.ZipFile, img_path: str):
    with src_zip.open(img_path, "r") as f:
        ds = dcmread(f)
        f.close()
    print("WTF!!!",img_path)
    return ds.pixel_array


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
    return data
            
