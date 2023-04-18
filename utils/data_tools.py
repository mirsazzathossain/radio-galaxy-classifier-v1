# -*- coding: utf-8 -*-

"""
Assentials tools for astronomical data processing.

This module includes the tools needed to retrieve catalogs from the Vizier
and astronomical data from the SkyView. Additionally, it includes the tools
necessary to convert fits images to numpy arrays and then to PIL images. It
also includes several tools for image processing and machine learning,
including the option to use masks to mask images and use the dataloader to
calculate the mean and standard deviation of PyTorch datasets.

Functions:
    get_catalog: Get the catalog of the astronomical objects
    get_single_fits: Download a single FITS image from the SkyView
    get_fits_images: Download the FITS images from the SkyView
    get_filename: Get the filename of the catalog
    get_class_code: Get the class code of the catalog
    fits_to_png: Convert a FITS image to PNG
    fits_to_png_batch: Convert a batch of FITS images to PNG
    mask_image: Mask the image with the given mask
    get_mean_std: Get the mean and standard deviation of the dataset
"""

__author__ = "Mir Sazzat Hossain"

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import torch
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from PIL import Image


def get_catalog(catalog_name: str) -> pd.DataFrame:
    """
    Get the catalog of the astronomical objects.

    :param catalog_name: Name of the catalog to be retrieved
    :type catalog_name: str

    :return: Catalog of the astronomical objects
    :rtype: pd.DataFrame
    """
    Vizier.ROW_LIMIT = -1
    catalog = Vizier.get_catalogs_async(catalog_name)[0]
    catalog = catalog.to_pandas()

    return catalog


def get_single_fits(
    survey: str,
    right_ascension: SkyCoord,
    declination: SkyCoord,
    file_name: str
) -> None:
    """
    Download a single FITS image from the SkyView.

    :param survey: Name of the astronomical survey e.g. "DSS2 Red"
    :type survey: str
    :param right_ascension: Right ascension of the astronomical object
    :type right_ascension: SkyCoord
    :param declination: Declination of the astronomical object
    :type declination: SkyCoord
    :param file_name: Path to the file to save the FITS image
    :type file_name: str
    """
    image = SkyView.get_images(
        position=str(right_ascension) + ", " + str(declination),
        survey=survey,
        coordinates="J2000",
        pixels=(150, 150),
    )[0]

    comment = str(image[0].header["comment"])
    comment = comment.replace("\n", "")
    comment = comment.replace("\t", " ")

    image[0].header.remove("comment", comment, True)
    image[0].header.add_comment(comment)

    folder_path = Path(file_name).parent
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    image.writeto(file_name, overwrite=True)


def get_filename(catalog: pd.DataFrame) -> str:
    """
    Get the filename of the catalog.

    :param catalog: Catalog of the astronomical objects
    :type catalog: pd.DataFrame

    :return: Filename of the catalog
    :rtype: str
    """
    if {"RAJ2000", "DEJ2000"}.issubset(catalog.keys()):
        filename = str(catalog["RAJ2000"])
        filename += "+" if catalog["DEJ2000"] > 0 else ""
        filename += str(catalog["DEJ2000"])
    elif {"RA", "Dec"}.issubset(catalog.keys()):
        if isinstance(catalog["RA"], str) and isinstance(catalog["Dec"], str):
            filename = catalog["RA"] + catalog["Dec"]
        else:
            filename = str(catalog["RA"]) + ("+" if catalog["Dec"] > 0 else "")
            filename += str(catalog["Dec"])
    elif "filename" in catalog:
        filename = catalog["filename"]
    elif "FCG" in catalog:
        filename = catalog["FCG"]
    else:
        raise Exception("No name found in catalog")

    return filename


def get_class_code(catalog: pd.DataFrame, classes: dict, column: str) -> str:
    """
    Get the class code of the catalog.

    :param catalog: Catalog of the astronomical objects
    :type catalog: pd.DataFrame
    :param classes: Dictionary of the classes to be used
    :type classes: dict
    :param column: Column name to get the class code
    :type column: str

    :return: Class code of the catalog
    :rtype: str
    """
    class_code = ""

    for key, value in classes.items():
        if column in catalog:
            if catalog[column].find(key) != -1:
                class_code = value
        else:
            raise Exception(
                "Column " + column + " not found in catalog " + str(catalog)
            )

    return class_code


def get_fits_images(
    catalog: pd.DataFrame,
    survey: str, save_dir: str,
    classes=None, column=None
) -> None:
    """
    Download the FITS images from the SkyView.

    :param catalog: Catalog of the astronomical objects
    :type catalog: pd.DataFrame
    :param survey: Name of the astronomical survey e.g. "DSS2 Red"
    :type survey: str
    :param save_dir: Path to the directory to save the FITS images
    :type save_dir: str
    :param classes: Dictionary of the classes to be used
    :type classes: dict
    :param column: Column name to get the class code
    :type column: str
    """
    failed = pd.DataFrame(columns=catalog.columns)

    for i in range(len(catalog)):
        try:
            name = get_filename(catalog.iloc[i])

            coordinate = SkyCoord(name, unit=(u.hourangle, u.deg))

            if not (coordinate.ra is not None and coordinate.dec is not None):
                raise AssertionError
            right_ascension = coordinate.ra.deg
            declination = coordinate.dec.deg

            if classes is not None and column is not None:
                class_code = get_class_code(catalog.iloc[i], classes, column)
            else:
                class_code = ""

            if "filename" in catalog:
                file_name = \
                    f"{save_dir}/{class_code}_{catalog['filename'][i]}.fits"
            else:
                file_name = f"{save_dir}/{class_code}_{name}.fits"

            get_single_fits(survey, right_ascension, declination, file_name)
        except Exception as exception:
            series = catalog.iloc[i].to_frame().T
            failed = pd.concat([failed, series], ignore_index=True)
            print(exception)


def fits_to_png(fits_path: str, im_size=None) -> Image.Image:
    """
    Convert a FITS image to PNG.

    :param fits_path: Path to the FITS image
    :type fits_path: str
    :param im_size: Size of the image
    :type im_size: tuple

    :return: Image in PNG format
    :rtype: Image.Image
    """
    try:
        img = fits.getdata(fits_path)
        header = fits.getheader(fits_path)
    except OSError:
        print("File not found: ", fits_path)
        return None

    if im_size is not None:
        width, height = im_size
    else:
        width, height = header["NAXIS1"], header["NAXIS2"]

    img = np.reshape(img, (height, width))

    # replace nan with nanmin
    img[np.isnan(img)] = np.nanmin(img)

    # make the pixel values between 0 and 255
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, mode="L")

    return img


def fits_to_png_batch(fits_dir: str, save_dir: str, im_size=None) -> None:
    """
    Convert a batch of FITS images to PNG.

    :param fits_dir: Path to the directory containing the FITS images
    :type fits_dir: str
    :param save_dir: Path to the directory to save the PNG images
    :type save_dir: str
    :param im_size: Size of the image
    :type im_size: tuple
    """
    for file in os.listdir(fits_dir):
        if file.endswith(".fits"):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            fits_path = os.path.join(fits_dir, file)
            png_path = os.path.join(save_dir, file.replace(".fits", ".png"))

            img = fits_to_png(fits_path, im_size)

            if img is not None:
                img.save(png_path)


def dataframe_to_html(catalog: pd.DataFrame, save_dir: str) -> None:
    """
    Save the catalog as an HTML file.

    :param catalog: Catalog of the astronomical objects
    :type catalog: pd.DataFrame
    :param save_dir: Path to the directory to save the HTML file
    :type save_dir: str
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    catalog.to_html(os.path.join(save_dir, "catalog.html"))


def mask_single_image(
    png_image: Image.Image,
    mask_image: Image.Image
) -> Image.Image:
    """
    Mask a single image with the mask.

    :param png_image: Image in PIL format
    :type png_image: Image.Image
    :param mask_image: Mask image in PIL format
    :type mask_image: Image.Image

    :return: Masked image in PIL format
    :rtype: Image.Image
    """
    png_img = np.array(png_image)
    mask_img = np.array(mask_image)

    png_img[mask_img == 0] = 0
    masked_image = Image.fromarray(png_img, mode="L")

    return masked_image


def mask_images(png_dir: str, mask_dir: str, save_dir: str) -> None:
    """
    Mask all the images in a directory with the mask.

    :param png_dir: Path to the directory containing the images
    :type png_dir: str
    :param mask_dir: Path to the directory containing the masks
    :type mask_dir: str
    :param save_dir: Path to the directory to save the masked images
    :type save_dir: str
    """
    for file in os.listdir(mask_dir):
        if file.endswith(".png"):
            png_file = os.path.join(png_dir, file)
            mask_file = os.path.join(mask_dir, file.replace(".png", ".png"))

            png_image = Image.open(png_file)
            mask_image = Image.open(mask_file)
            masked_image = mask_single_image(png_image, mask_image)

            Path(save_dir).mkdir(parents=True, exist_ok=True)
            masked_image.save(os.path.join(save_dir, file))


def get_mean_and_std(dataloader: torch.utils.data.DataLoader) -> tuple:
    """
    Compute the mean and standard deviation of the dataset.

    :param dataloader: Dataloader of the dataset
    :type dataloader: torch.utils.data.DataLoader

    :return: Mean and standard deviation of the dataset
    :rtype: tuple
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def clear_folder(folder: str, extension: list) -> None:
    """
    Clear all files in a folder except the ones with the given extensions.

    :param folder: Path to the folder to clear
    :type folder: str
    :param extension: List of extensions to keep
    :type extension: list
    """
    for file in os.listdir(folder):
        if not file.endswith(tuple(extension)):
            os.remove(os.path.join(folder, file))

    print(f"Folder {folder} cleared.")
