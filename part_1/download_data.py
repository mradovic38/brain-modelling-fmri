import os
import requests

fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]
urls = [
    "https://osf.io/r638s/download",
    "https://osf.io/yqb3e/download",
    "https://osf.io/ymnjv/download"
]


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    for fname, url in zip(fnames, urls):
        path = os.path.join(data_dir, fname)

        if not os.path.isfile(path):
            try:
                r = requests.get(url)
                r.raise_for_status()
            except requests.RequestException:
                print("!!! Failed to download data !!!")
            else:
                print(f"Downloading {fname}...")
                with open(path, "wb") as fid:
                    fid.write(r.content)
                print(f"Download {fname} completed!")

    fname = os.path.join('data', fname)


if __name__ == '__main__':
    main()