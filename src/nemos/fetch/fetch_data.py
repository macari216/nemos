"""
Module for fetching data and utilities using the pooch library.

This module allows you to download datasets and utility scripts
from specified URLs and manage their versions and integrity using
SHA256 hash verification. It also provides helper functions to
calculate hashes for local files and manage file paths.
"""

import hashlib
import os
import pathlib
from typing import List, Optional, Union

try:
    import pooch
    from pooch import Pooch
except ImportError:
    pooch = None
    Pooch = None

try:
    import dandi
    import fsspec
    import h5py
    from dandi.dandiapi import DandiAPIClient
    from fsspec.implementations.cached import CachingFileSystem
    from pynwb import NWBHDF5IO
except ImportError:
    dandi = None
    NWBHDF5IO = None


# Registry of dataset filenames and their corresponding SHA256 hashes.
REGISTRY_DATA = {
    "A0670-221213.nwb": "8587dd6dde107504bd4a17a68ce8fb934fcbcccc337e653f31484611ee51f50a",
    "Mouse32-140822.nwb": "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35",
    "Achilles_10252013.nwb": "42857015aad4c2f7f6f3d4022611a69bc86d714cf465183ce30955731e614990",
    "allen_478498617.nwb": "262393d7485a5b39cc80fb55011dcf21f86133f13d088e35439c2559fd4b49fa",
    "m691l1.nwb": "1990d8d95a70a29af95dade51e60ffae7a176f6207e80dbf9ccefaf418fe22b6",
}
DOWNLOADABLE_FILES = list(REGISTRY_DATA.keys())

# URL templates for downloading datasets and utility scripts.
OSF_TEMPLATE = "https://osf.io/{}/download"

# Mapping of dataset filenames to their download URLs.
REGISTRY_URLS_DATA = {
    "A0670-221213.nwb": OSF_TEMPLATE.format("sbnaw"),
    "Mouse32-140822.nwb": OSF_TEMPLATE.format("jb2gd"),
    "Achilles_10252013.nwb": OSF_TEMPLATE.format("hu5ma"),
    "allen_478498617.nwb": OSF_TEMPLATE.format("vf2nj"),
    "m691l1.nwb": OSF_TEMPLATE.format("xesdm"),
}

_NEMOS_ENV = "NEMOS_DATA_DIR"


def _calculate_sha256(data_dir: Union[str, pathlib.Path]):
    """
    Calculate the SHA256 hash for each file in a directory.

    This function iterates through files in the specified directory
    and computes their SHA256 hash. This is useful for verifying
    file integrity or generating new registry entries.

    Parameters
    ----------
    data_dir :
        The path to the directory containing the files to hash.

    Returns
    -------
    :
        A dictionary where the keys are filenames and the values
        are their corresponding SHA256 hashes.
    """
    data_dir = pathlib.Path(data_dir)

    # Initialize the registry dictionary to store file hashes.
    registry_hash = dict()
    for file_path in data_dir.iterdir():
        if file_path.is_dir():
            continue
        # Open the file in binary mode to read it.
        with open(file_path, "rb") as f:
            sha256_hash = hashlib.sha256()  # Initialize the SHA256 hash object.
            # Read the file in chunks to avoid loading it all into memory.
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)  # Update the hash with the chunk.
            registry_hash[file_path.name] = sha256_hash.hexdigest()  # Store the hash.
    return registry_hash


def _create_retriever(path: Optional[pathlib.Path] = None) -> Pooch:
    """Create a pooch retriever for fetching datasets.

    This function sets up the pooch retriever, which manages the
    downloading and caching of files, including handling retries
    and checking file integrity using SHA256 hashes.

    Parameters
    ----------
    path :
        The directory where datasets will be stored. If not provided,
        defaults to pooch's cache (check ``pooch.os_cache('nemos')`` for that path)

    Returns
    -------
    :
        A configured pooch retriever object.

    """
    if path is None:
        # Use the default data directory if none is provided.
        path = pooch.os_cache("nemos")

    return pooch.create(
        path=path,
        base_url="",
        urls=REGISTRY_URLS_DATA,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
        env=_NEMOS_ENV,
    )


def _find_shared_directory(paths: List[pathlib.Path]) -> pathlib.Path:
    """
    Find the common parent directory shared by all given paths.

    This function takes a list of file paths and determines the
    highest-level directory that all paths share.

    Parameters
    ----------
    paths :
        A list of file paths.

    Returns
    -------
    :
        The shared parent directory.

    Raises
    ------
    ValueError
        If no paths are provided or if the paths do not share a common directory.
    """
    # Iterate through the parents of the first path to find a common directory.
    if len(paths) == 0:
        raise ValueError(
            "Must provide at least one path. The input list of paths is empty."
        )

    if len(paths[0].parents) == 0:
        raise ValueError("The provided path does not have any parent directories.")

    for directory in paths[0].parents:
        if all([directory in p.parents for p in paths]):
            return directory

    raise ValueError("The provided paths do not share a common parent directory.")


def fetch_data(
    dataset_name: str, path: Optional[Union[pathlib.Path, str]] = None
) -> str:
    """
    Download a dataset using pooch.

    This function downloads a dataset, checking if it already exists
    and is unchanged. If the dataset is an archive (ends in .tar.gz),
    it decompresses the archive and returns the path to the resulting
    directory. Otherwise, it returns the path to the downloaded file.

    Parameters
    ----------
    dataset_name :
        The name of the dataset to download. Must match an entry in
        REGISTRY_DATA.
    path :
        The directory where the dataset will be stored. If not provided,
        defaults to _DEFAULT_DATA_DIR.

    Returns
    -------
    :
        The path to the downloaded file or directory.
    """
    if pooch is None:
        raise ImportError(
            "Missing optional dependency 'pooch'."
            " Please use pip or "
            "conda to install 'pooch'."
        )
    retriever = _create_retriever(path)
    return _retrieve_data(dataset_name, retriever).as_posix()


def _retrieve_data(dataset_name: str, retriever: Pooch) -> pathlib.Path:
    """
    Helper function to fetch and process a dataset.

    This function is used internally to download a dataset and, if
    necessary, decompress it.

    Parameters
    ----------
    dataset_name :
        The name of the dataset to download.
    retriever :
        The pooch retriever object used to fetch the dataset.

    Returns
    -------
    :
        The path to the downloaded file or directory.
    """
    # Determine if the dataset is an archive and set the appropriate processor.
    if dataset_name.endswith(".tar.gz"):
        processor = pooch.Untar()
    else:
        processor = None

    # Fetch the dataset using pooch.
    file_name = retriever.fetch(dataset_name, progressbar=True, processor=processor)

    # If the dataset was an archive, find the shared directory; otherwise, return the file path.
    if dataset_name.endswith(".tar.gz"):
        file_name = _find_shared_directory([pathlib.Path(f) for f in file_name])
    else:
        file_name = pathlib.Path(file_name)

    return file_name


def download_dandi_data(dandiset_id: str, filepath: str) -> NWBHDF5IO:
    """Download a dataset from the DANDI Archive (https://dandiarchive.org/)

    Parameters
    ----------
    dandiset_id :
        6-character string of numbers giving the ID of the dandiset.
    filepath :
        filepath to the specific .nwb file within the dandiset we wish to return.

    Returns
    -------
    io :
        NWB file containing specified data.

    Examples
    --------
    >>> import nemos as nmo
    >>> io = nmo.fetch.download_dandi_data("000582",
                                           "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb")
    >>> nwb = nap.NWBFile(io.read(), lazy_loading=False)
    >>> print(nwb)

    """
    if dandi is None:
        raise ImportError(
            "Missing optional dependency 'dandi'."
            " Please use pip or "
            "conda to install 'dandi'."
        )
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # first, create a virtual filesystem based on the http protocol
    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    # mimicking caching behavior of pooch create
    if _NEMOS_ENV in os.environ:
        cache_dir = pathlib.Path(os.environ[_NEMOS_ENV])
    else:
        cache_dir = pooch.os_cache("nemos") / "nwb-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    fs = CachingFileSystem(
        fs=fs,
        cache_storage=cache_dir.as_posix(),  # Local folder for the cache
    )

    # next, open the file
    file = h5py.File(fs.open(s3_url, "rb"))
    io = NWBHDF5IO(file=file, load_namespaces=True)

    return io
