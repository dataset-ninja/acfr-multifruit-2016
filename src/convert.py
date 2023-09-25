# https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/

import csv
import glob
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    
    # project_name = "ACFR Orchard Fruit"
    dataset_path = "/home/grokhi/rawdata/acfr-multifruit-2016/acfr-fruit-dataset"
    images_folder = "images"
    bboxes_folder = "annotations"
    masks_folder = "segmentations"
    images_ext = ".png"
    masks_ext = "_L.png"
    bboxes_ext = ".csv"
    batch_size = 30


    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        obj_class = name_to_class[subfolder]

        image_name = get_file_name(image_path)

        # bad quality for masks
        # if subfolder == "apples":
        #     mask_path = os.path.join(masks_path, image_name + masks_ext)
        #     ann_np = sly.imaging.image.read(mask_path)[:, :, 0]
        #     obj_mask = ann_np == 128
        #     ret, curr_mask = connectedComponents(obj_mask.astype("uint8"), connectivity=8)
        #     if ret > 1:
        #         for i in range(1, ret):
        #             obj_mask = curr_mask == i
        #             curr_bitmap = sly.Bitmap(obj_mask)
        #             curr_label = sly.Label(curr_bitmap, obj_class)
        #             labels.append(curr_label)

        box_path = os.path.join(bboxes_path, image_name + bboxes_ext)
        with open(box_path, "r") as file:
            csvreader = csv.reader(file)
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                if subfolder == "apples":
                    top = float(row[2]) - float(row[3])
                    left = float(row[1]) - float(row[3])
                    bottom = float(row[2]) + float(row[3])
                    right = float(row[1]) + float(row[3])
                else:
                    top = float(row[2])
                    left = float(row[1])
                    bottom = top + float(row[4])
                    right = left + float(row[3])
                rect = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                curr_label = sly.Label(rect, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    almond = sly.ObjClass("almond", sly.Rectangle)
    mango = sly.ObjClass("mango", sly.Rectangle)
    apple = sly.ObjClass("apple", sly.AnyGeometry)

    name_to_class = {"almonds": almond, "mangoes": mango, "apples": apple}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[almond, apple, mango])
    api.project.update_meta(project.id, meta.to_json())


    train_pathes = glob.glob(dataset_path + "/*/sets/train.txt")
    ds_to_train_names = defaultdict(list)
    for train_path in train_pathes:
        curr_ds = train_path.split("/")[-3]
        with open(train_path) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) > 0:
                    ds_to_train_names[curr_ds].append(curr_data + images_ext)

    val_pathes = glob.glob(dataset_path + "/*/sets/val.txt")
    ds_to_val_names = defaultdict(list)
    for val_path in val_pathes:
        curr_ds = val_path.split("/")[-3]
        with open(val_path) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) > 0:
                    ds_to_val_names[curr_ds].append(curr_data + images_ext)

    test_pathes = glob.glob(dataset_path + "/*/sets/test.txt")
    ds_to_test_names = defaultdict(list)
    for test_path in test_pathes:
        curr_ds = test_path.split("/")[-3]
        with open(test_path) as f:
            content = f.read().split("\n")
            for curr_data in content:
                if len(curr_data) > 0:
                    ds_to_test_names[curr_ds].append(curr_data + images_ext)

    ds_name_to_im_names = {"train": ds_to_train_names, "val": ds_to_val_names, "test": ds_to_test_names}

    for ds_name, im_names_data in ds_name_to_im_names.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for subfolder in im_names_data.keys():
            images_path = os.path.join(dataset_path, subfolder, images_folder)
            bboxes_path = os.path.join(dataset_path, subfolder, bboxes_folder)
            if subfolder == "apples":
                masks_path = os.path.join(dataset_path, subfolder, masks_folder)
            images_names = im_names_data[subfolder]

            progress = sly.Progress(
                "Create dataset {}, add {} data".format(ds_name, subfolder), len(images_names)
            )

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [os.path.join(images_path, im_name) for im_name in img_names_batch]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))
    return project


