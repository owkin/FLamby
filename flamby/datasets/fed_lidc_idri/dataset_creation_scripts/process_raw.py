import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import dicom_numpy
import networkx as nx
import nibabel as nib
import numpy as np
import pydicom
from scipy.spatial import cKDTree
from skimage.draw import polygon
from skimage.transform import resize as sk_resize


def render_nodule_3D(nodule, shape):
    """
    Render a nodule into 3D
    Parameters
    ----------
    nodule : list
        Must be a list of rois, each roi begin a 2darray: [[xs], [ys], [z]]
    shape : tuple
        The shape in which to render
    Returns
    -------
    ndarray
        the boolean mask
    """
    mask = np.zeros(shape, np.uint8)

    for roi in nodule:
        z = roi[-1, 0]
        minx, miny = roi[:2].min(axis=1)
        maxx, maxy = roi[:2].max(axis=1)
        xs, ys = roi[:2]

        xs, ys = polygon(xs - minx, ys - miny, shape=(maxx - minx, maxy - miny))
        region = tuple([xs + minx, ys + miny])
        mask[..., z][region] = 1

    return mask


def saferesize(arr, affine, spacing=[1.0, 1.0, 1.0], dicomdir=None):
    """
    Resize an array along with its affine. Takes care of integer-valued arrays
    and interpolation artifacts.
    Parameters
    ----------
    arr : ndarray
        array to resize
    affine : ndarray
        affin amtrix, 4x4
    spacing : list, optional
        The desired spacing in mm
    Returns
    -------
    ndarray
        the resized array
    ndarray
        the corrected affine
    """
    spacing = np.array(spacing)
    old_spacing = np.abs(affine.diagonal()[:-1])

    new_shape = np.round(np.array(arr.shape) * old_spacing / spacing).astype(int)

    if arr.dtype != np.uint8:
        new_arr = sk_resize(
            arr, new_shape, preserve_range=True, anti_aliasing=False, mode="constant"
        )
    else:
        new_arr = sk_resize(
            (arr == 1).astype(np.float32),
            new_shape,
            preserve_range=True,
            anti_aliasing=False,
            mode="constant",
        )

        new_arr = new_arr.round().astype(np.uint8)

        twos = (
            np.stack(np.where(arr == 2), 0)
            * (new_shape / np.array(arr.shape))[..., np.newaxis]
        )
        threes = (
            np.stack(np.where(arr == 3), 0)
            * (new_shape / np.array(arr.shape))[..., np.newaxis]
        )

        new_arr[tuple(twos.round().astype(int))] = 2
        new_arr[tuple(threes.round().astype(int))] = 3

    new_affine = affine.copy()
    new_affine[:3, -1] *= new_affine.diagonal()[:-1] / spacing
    new_affine[:3, :3] = np.diag(spacing)

    return new_arr, new_affine


def convert_to_niftis(dicomdir, xml_path, spacing=None, force=False):
    """
    Core function, writes down to NiFTIs and eventually resamples.
    Parameters
    ----------
    dicomdir : path
        path to the directory containing extracted dicoms
    xml_path : path
        path to the annotation xml file
    spacing : list of floats, optional
        Description
    average_masks : bool, optional
        Description
    force
    """

    if (
        force
        or not os.path.exists(dicomdir + "/patient.nii.gz")
        or not os.path.exists(dicomdir + "/mask.nii.gz")
        or not os.path.exists(dicomdir + "/mask_consensus.nii.gz")
    ):
        try:
            ct, masks, affine = build_volumes(dicomdir, xml_path)
        except FileNotFoundError:
            raise
        except dicom_numpy.DicomImportException as e:
            print(f"\n{e} in file {dicomdir}, xml {xml_path}). Skipping it.")
            return False

        if affine is not None and ct is not None and masks:
            if spacing is not None:
                masks = [
                    saferesize(mask, affine, dicomdir=dicomdir)[0] for mask in masks
                ]
                ct, affine = saferesize(ct, affine)

            CT = nib.Nifti1Image(ct.astype(np.float32), affine)
            nib.save(CT, dicomdir + "/patient.nii.gz")

            masks = np.stack(masks, -1)

            mask_consensus = build_consensus(masks)
            MASK_CONSENSUS = nib.Nifti1Image(mask_consensus, affine)
            nib.save(MASK_CONSENSUS, dicomdir + "/mask_consensus.nii.gz")

            MASK = nib.Nifti1Image(masks, affine)
            nib.save(MASK, dicomdir + "/mask.nii.gz")

            return True
        else:
            print(
                "Volumes were not correctly built. build_volumes returned 'affine'",
                "'ct' and/or 'Issue' as None."
                "Unknown error at dicomdir %s, xml %s" % (dicomdir, xml_path),
            )
            return False
    else:
        return True


def build_consensus(masks):
    """
    Average the various masks. Takes care of the point annotations by clustering them.
    Parameters
    ----------
    masks : TYPE
        Description
    """

    # TODO: what happens when one says big nodule (mask) and the other one
    # says point (inside the mask)
    # mask = np.zeros_like(masks[0])

    # first, average the nodule segmented:
    if masks.size == 0:
        mask = np.empty((0, 0, 0), np.uint8)
    else:
        mask = ((masks == 1).sum(-1) > 0.0).astype(np.uint8)

    # Then cluster nodules and non nodules
    nodules = np.stack(np.where((mask == 2).sum(-1) > 0), 0).T
    if nodules.shape[0] > 1:
        clustered_nodules = fuse_points(nodules)
        mask[tuple(clustered_nodules.T)] = 2

    nonnodules = np.stack(np.where((mask == 3).sum(-1) > 0), 0).T
    if nonnodules.shape[0] > 1:
        clustered_nonnodules = fuse_points(nonnodules)
        mask[tuple(clustered_nonnodules.T)] = 3

    return mask


def fuse_points(nonNodules, radius=15):
    """Summary
    Parameters
    ----------
    nonNodules : TYPE
        Description
    radius : int, optional
        Description
    Returns
    -------
    TYPE
        Description
    """
    tree = cKDTree(nonNodules)
    pairs = tree.query_pairs(radius / 2)
    if len(pairs) > 0:
        g = nx.Graph()

        for i, non in enumerate(nonNodules):
            g.add_node(i)

        for p in pairs:
            g.add_edge(*p)

        fused = []
        for sg in nx.connected_component_subgraphs(g):
            those_nodes = list(sg.nodes)
            fuse = nonNodules[those_nodes].mean(0).round().astype(int)
            fused.append(fuse)

        return np.stack(fused, 0)

    else:
        return nonNodules


def correct_ct_intensity(data):
    if data.min() < -1.0:  # -1 just for safety but should be 0
        data = np.clip(data, -1024.0, None)
    if data.min() > -1.0:
        data += -1024.0

    if data.mean() > -50.0:
        data -= 1024.0
        data = np.clip(data, a_min=-1024.0, a_max=None)
    return data


def build_volumes(dicomdir, xml_path):
    """
    Reconstructs the CT scan volume and the annotations from the
    dicom files and the corresponding annotation file.
    Parameters
    ----------
    dicomdir : path
        path to directory containing dicoms.
    xml_path : path
        path to annotation file
    Returns
    -------
    ndarray
        CT scan
    list
        list of ndarray (for each radiologist)
    """
    dcms = glob.glob(dicomdir + "/*.dcm")
    dicoms = [pydicom.read_file(dcm, stop_before_pixels=False) for dcm in dcms]
    dicoms = sorted(dicoms, key=lambda x: float(x.ImagePositionPatient[-1]))
    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(dicoms)
    voxel_ndarray = correct_ct_intensity(voxel_ndarray)

    def opNN(nonNodule):
        z = float(nonNodule.findtext("{http://www.nih.gov}imageZposition"))

        # pylidc trick !
        locz = np.abs(
            [z - float(dicom.ImagePositionPatient[-1]) for dicom in dicoms]
        ).argmin()

        locx = int(
            nonNodule.findtext("{http://www.nih.gov}locus/{http://www.nih.gov}xCoord")
        )
        locy = int(
            nonNodule.findtext("{http://www.nih.gov}locus/{http://www.nih.gov}yCoord")
        )

        out = np.array([locy, locx, locz]).reshape(-1, 1)

        return out.astype(np.int64)

    def opN(Nodule):
        xyzs = []

        for roi in Nodule.iter("{http://www.nih.gov}roi"):
            z = float(roi.findtext("{http://www.nih.gov}imageZposition"))
            locz = np.abs(
                [z - float(dicom.ImagePositionPatient[-1]) for dicom in dicoms]
            ).argmin()

            xs = roi.findall("{http://www.nih.gov}edgeMap/{http://www.nih.gov}xCoord")
            ys = roi.findall("{http://www.nih.gov}edgeMap/{http://www.nih.gov}yCoord")
            zs = [locz] * len(xs)

            xs = np.array([int(x.text) for x in xs])
            ys = np.array([int(y.text) for y in ys])
            zs = np.array(zs)

            xyzs.append(np.array([xs, ys, zs]).astype(np.int64))

        return xyzs

    root = ET.parse(xml_path).getroot()

    masks = []

    for i, rs in enumerate(root.iter("{http://www.nih.gov}readingSession")):
        this_session_nodules = [
            opN(nodule) for nodule in rs.iter("{http://www.nih.gov}unblindedReadNodule")
        ]
        this_session_nonnodules = [
            opNN(nonnodule) for nonnodule in rs.iter("{http://www.nih.gov}nonNodule")
        ]

        mask = np.zeros(voxel_ndarray.shape, np.uint8)

        try:
            for nodule in this_session_nodules:
                if len(nodule) > 1:
                    mask += render_nodule_3D(nodule, shape=mask.shape)
                else:
                    mask[tuple(nodule[0])] = 2

            for nonnodule in this_session_nonnodules:
                mask[tuple(nonnodule)] = 3
        except IndexError:
            print("Indexing issue at file %s, session %i" % (xml_path, i))
            pass
        masks.append(mask)

    return voxel_ndarray, masks, ijk_to_xyz


def clean_up_dicoms(dicomdir):
    """
    Recursively remove all DICOM files from a folder and its subfolders.
    Parameters
    ----------
    dicomdir : path
        path to directory containing dicoms.
    """
    for dicom in Path(dicomdir).rglob("*.dcm"):
        os.remove(dicom)
