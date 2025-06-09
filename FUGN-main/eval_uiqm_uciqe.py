import os
import cv2
import numpy as np
import argparse
from uiqm import getUIQM, getUCIQE
from tqdm import tqdm
import pandas as pd


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif'])


def evaluate_folder(image_dir, save_csv_path=None):
    uiqm_list = []
    uciqe_list = []
    filenames = []

    image_files = [f for f in os.listdir(image_dir) if is_image_file(f)]
    image_files.sort()

    for fname in tqdm(image_files, desc="Evaluating UIQM/UCIQE"):
        img_path = os.path.join(image_dir, fname)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[Warning] Could not read image: {fname}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            uiqm_score = getUIQM(image_rgb)
            uciqe_score = getUCIQE(image_rgb)
        except Exception as e:
            print(f"[Error] Failed on {fname}: {e}")
            continue

        filenames.append(fname)
        uiqm_list.append(uiqm_score)
        uciqe_list.append(uciqe_score)

    print("\n==== Average Results ====")
    print(f"Average UIQM : {np.mean(uiqm_list):.4f}")
    print(f"Average UCIQE: {np.mean(uciqe_list):.4f}")

    if save_csv_path:
        df = pd.DataFrame({
            "Filename": filenames,
            "UIQM": uiqm_list,
            "UCIQE": uciqe_list
        })
        df.to_csv(save_csv_path, index=False)
        print(f"[Info] Saved results to: {save_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to folder containing enhanced images (e.g., result/net_C/test_no_ref/)')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Path to save CSV result (optional)')
    args = parser.parse_args()

    evaluate_folder(args.image_dir, args.save_csv)
