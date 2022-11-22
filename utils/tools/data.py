"""
Assentials tools for astronomical data processing.

This module includes the tools needed to retrieve catalogs from the Vizier
and astronomical data from the SkyView. Additionally, it includes the tools
necessary to convert fits images to numpy arrays and then to PIL images. It
also includes several tools for image processing and machine learning, including
the option to use masks to mask images and use the dataloader to calculate the
mean and standard deviation of PyTorch datasets.

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
    Get the catalog of the astronomical objects

    Arguments:
        catalog_name {str}: Name of the catalog as in Vizier

    Returns:
        pd.DataFrame: Catalog of the astronomical objects
    """
    Vizier.ROW_LIMIT = -1
    catalog = Vizier.get_catalogs_async(catalog_name)[0]  # type: ignore
    catalog = catalog.to_pandas()  # type: ignore

    return catalog


def get_single_fits(
    survey: str, right_ascension: SkyCoord, declination: SkyCoord, file_name: str
) -> None:
    """
    Download a single FITS image from the SkyView

    Arguments:
        survey {str}: Name of the astronomical survey e.g. "DSS2 Red"
        right_ascension {SkyCoord}: Right Ascension of the object
        declination {SkyCoord}: Declination of the object
        file_name {str}: Name of the FITS file to be saved
    """
    image = SkyView.get_images(
        position=str(right_ascension) + ", " + str(declination),
        survey=survey,
        coordinates="J2000",
        pixels=(150, 150),
    )[0]

    comment = str(image[0].header["comment"])  # type: ignore
    comment = comment.replace("\n", "")
    comment = comment.replace("\t", " ")

    image[0].header.remove("comment", comment, True)  # type: ignore
    image[0].header.add_comment(comment)  # type: ignore

    folder_path = Path(file_name).parent
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    image.writeto(file_name, overwrite=True)


def get_filename(catalog: pd.DataFrame) -> str:
    """
    Get the filename of the catalog

    Arguments:
        catalog {pd.DataFrame}: Catalog of the astronomical objects

    Returns:
        str: Filename of the catalog
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

    return filename  # type: ignore


def get_class_code(catalog: pd.DataFrame, classes: dict, column: str) -> str:
    """
    Get the class code of the catalog

    Arguments:
        catalog {pd.DataFrame}: Catalog of the astronomical objects
        classes {dict}: Dictionary of the classes to be used
        column {str}: Column name to get the class code

    Returns:
        str: Class code of the catalog
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
    catalog: pd.DataFrame, survey: str, save_dir: str, classes=None, column=None
) -> None:
    """
    Download the FITS images from the SkyView

    Arguments:
        catalog {pd.DataFrame}: Catalog of the astronomical objects
        survey {str}: Name of the astronomical survey e.g. "DSS2 Red"
        save_dir {str}: Path to the directory to save the FITS images
        classes {dict}: Dictionary of the classes to be used
        column {str}: Column name to get the class code
    """
    failed = pd.DataFrame(columns=catalog.columns)

    for i in range(len(catalog)):
        try:
            name = get_filename(catalog.iloc[i])  # type: ignore

            coordinate = SkyCoord(name, unit=(u.hourangle, u.deg))

            assert coordinate.ra is not None and coordinate.dec is not None
            right_ascension = coordinate.ra.deg
            declination = coordinate.dec.deg

            if classes is not None and column is not None:
                class_code = get_class_code(
                    catalog.iloc[i], classes, column)  # type: ignore
            else:
                class_code = ""

            if "filename" in catalog:
                file_name = f"{save_dir}/{class_code}_{catalog['filename'][i]}.fits"
            else:
                file_name = f"{save_dir}/{class_code}_{name}.fits"

            get_single_fits(survey, right_ascension,
                            declination, file_name)  # type: ignore
        except Exception as exception:  # pylint: disable=broad-except
            series = catalog.iloc[i].to_frame().T
            failed = pd.concat([failed, series], ignore_index=True)
            print(exception)


def fits_to_png(fits_path: str, im_size=None) -> Image.Image:
    """
    Convert a FITS image to PNG

    Arguments:
        fits_path {str}: Path to the FITS image
        png_path {str}: Path to the PNG image
        im_size {tuple}: Size of the image
    """
    try:
        img = fits.getdata(fits_path)
        header = fits.getheader(fits_path)
    except OSError:
        print("File not found: ", fits_path)
        return None  # type: ignore

    if im_size is not None:
        width, height = im_size
    else:
        width, height = header["NAXIS1"], header["NAXIS2"]

    img = np.reshape(img, (height, width))  # type: ignore

    # replace nan with nanmin
    img[np.isnan(img)] = np.nanmin(img)

    # make the pixel values between 0 and 255
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, mode="L")

    return img


def fits_to_png_batch(fits_dir: str, save_dir: str, im_size=None) -> None:
    """
    Convert a batch of FITS images to PNG

    Arguments:
        fits_dir {str}: Path to the FITS images directory
        save_dir {str}: Path to the directory to save the PNG images
        im_size {tuple}: Size of the image
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
    Save the catalog as an HTML file

    Arguments:
        catalog {pd.DataFrame}: Catalog to save
        save_dir {str}: Path to the directory to save the HTML file
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    catalog.to_html(os.path.join(save_dir, "catalog.html"))


def mask_single_image(png_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    """
    Mask a single image with the mask

    Arguments:
        png_file {str}: Path to the PNG image
        mask_file {str}: Path to the mask image

    Returns:
        masked_image {PIL.Image}: Masked image
    """
    png_img = np.array(png_image)
    mask_img = np.array(mask_image)

    png_img[mask_img == 0] = 0
    masked_image = Image.fromarray(png_img, mode="L")

    return masked_image


def mask_images(png_dir: str, mask_dir: str, save_dir: str) -> None:
    """
    Mask all the images in a directory with the mask

    Arguments:
        png_dir {str}: Path to the source images directory
        mask_dir {str}: Path to the mask images directory
        save_dir {str}: Path to the directory to save the masked images
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


def get_mean_and_std(dataloader: torch.utils.data.DataLoader) -> tuple:  # type: ignore
    """
    Compute the mean and standard deviation of the dataset

    Arguments:
        dataloader {torch.utils.data.DataLoader}: Dataloader of the dataset

    Returns:
        mean {float}: Mean of the dataset
        std {float}: Standard deviation of the dataset
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in dataloader:
        # Mean over batch, height and width, but not over the channels
        # pylint: disable=E1101
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        # pylint: disable=E1101
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def clear_folder(folder: str, extension: list) -> None:
    """
    Clear all files in a folder except the ones with the given extensions.

    Arguments:
        folder {str}: Path to the folder
        extension {list}: List of extensions to keep
    """
    for file in os.listdir(folder):
        if not file.endswith(tuple(extension)):
            os.remove(os.path.join(folder, file))

    print(f"Folder {folder} cleared.")
