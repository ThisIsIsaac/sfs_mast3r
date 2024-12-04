import h5py
import numpy as np
import json
import os
import sfs
from sfs_util import check_and_download_image
import json

num_imgs=1_000_000
url_file_name="random_urls_5M.json"
download_path = "/viscam/projects/sfs/mast3r/mast3r_outputs/random_urls_500k"
os.makedirs(download_path, exist_ok=True)
sfs.select_random_rows("/viscam/u/iamisaac/datacomp/small_merged/metadata.hdf5", num_imgs*5, url_file_name)

download_path = "/viscam/projects/sfs/mast3r/mast3r_outputs/random_imgs"
with open(url_file_name, "r") as file:
    urls = json.load(file)
num_failed = 0
for url in urls:
    try:
        check_and_download_image(url, download_path, set())
    except:
        num_failed +=1
print(f"num failed = {num_failed}")
    