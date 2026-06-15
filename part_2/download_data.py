import os
import requests
import tarfile
import numpy as np

fname = "data/hcp_task.tgz"
url = "https://osf.io/2y3fw/download"
HCP_DIR = "data/hcp_task"


def download():
    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)

def extract():
    with tarfile.open(fname) as tfile:
        tfile.extractall('.')


if __name__ == '__main__':
    download()
    extract()