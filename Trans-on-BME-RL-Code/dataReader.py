#!/usr/bin/env python
# -*- coding : utf-8-*-
# coding:unicode_escape
# File: dataReader.py

import SimpleITK as sitk
import numpy as np
import warnings

warnings.simplefilter("ignore", category=ResourceWarning)


__all__ = [
    'filesListBrainMRLandmark',
    'filesListCardioLandmark',
    'filesListFetalUSLandmark',
    'NiftiImage']


def getLandmarksFromTXTFile(file, split=','):
    """
    Extract each landmark point line by line from a text file, and return
    vector containing all landmarks.
    """
    with open(file, 'r', encoding='utf-8') as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(split)])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks

###############################################################################

class filesListBrainMRLandmark(object):
    """ A class for managing train files for mri brain data

        Attributes:
        files_list: Two or one text files that contain a list of all images and
        (landmarks)
        returnLandmarks: Return landmarks if task is train or eval
        (default: True)
    """
    def __init__(self, files_list=None, returnLandmarks=True, agents=1):
        # check if files_list exists
        assert files_list, 'There is no file given'
        # read image filenames
        self.image_files = [line.split('\n')[0]
                            for line in open(files_list[0].name)]
        # read landmark filenames if task is train or eval
        self.returnLandmarks = returnLandmarks
        self.agents = agents
        if self.returnLandmarks:
            self.landmark_files = [
                line.split('\n')[0] for line in open(
                    files_list[1].name)]
            assert len(
                self.image_files) == len(
                self.landmark_files), """number of image files is not equal to
                number of landmark files"""

    @property
    def num_files(self):
        return len(self.image_files)

    def sample_circular(self, landmark_ids, shuffle=False):
        """ return a random sampled ImageRecord from the list of files
        """

        if shuffle:
            # TODO: could use PyTorch shuffles
            # indexes = rng.choice(x, len(x), replace=False)
            pass
        else:
            indexes = np.arange(self.num_files)

        while True:
            for idx in indexes:
                sitk_image, image = NiftiImage().decode(self.image_files[idx])
                if self.returnLandmarks:
                    # transform landmarks to image space if they are in
                    # physical space
                    landmark_file = self.landmark_files[idx]
                    all_landmarks = getLandmarksFromTXTFile(landmark_file)
                    # landmark index is 13 for ac-point and 14 pc-point
                    # transform landmark from physical to image space if
                    # required
                    # landmarks = sitk_image.
                    #         TransformPhysicalPointToContinuousIndex(landmark)
                    landmarks = [np.round(all_landmarks[landmark_ids[i] % 15])
                                 for i in range(self.agents)]
                else:
                    landmarks = None
                # extract filename from path, remove .nii.gz extension
                image_filenames = [self.image_files[idx][:-7]] * self.agents
                images = [image] * self.agents
                yield (images, landmarks, image_filenames,
                       sitk_image.GetSpacing())


class ImageRecord(object):
    """image object to contain height,width, depth and name """
    pass


class NiftiImage(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        pass

    def _is_nifti(self, filename):
        """Determine if a file contains a nifti format image.
        Args
          filename: string, path of the image file
        Returns
          boolean indicating if the image is a nifti
        """
        extensions = ['.nii', '.nii.gz']
        return any(i in filename for i in extensions)

    def decode(self, filename, label=False):
        """ decode a single nifti image
        Args
          filename: string for input images
          label: True if nifti image is label
        Returns
          image: an image container with attributes; name, data, dims
        """
        image = ImageRecord()
        image.name = filename
        assert self._is_nifti(
            image.name), "unknown image format for %r" % image.name

        if label:
            sitk_image = sitk.ReadImage(image.name, sitk.sitkInt8)
        else:
            sitk_image = sitk.ReadImage(image.name, sitk.sitkFloat32)
            np_image = sitk.GetArrayFromImage(sitk_image)
            # threshold image between p10 and p98 then re-scale [0-255]
            p0 = np_image.min().astype('float')
            p10 = np.percentile(np_image, 10)
            p99 = np.percentile(np_image, 99)
            p100 = np_image.max().astype('float')
            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p10,
                                        upper=p100,
                                        outsideValue=p10)
            sitk_image = sitk.Threshold(sitk_image,
                                        lower=p0,
                                        upper=p99,
                                        outsideValue=p99)
            sitk_image = sitk.RescaleIntensity(sitk_image,
                                               outputMinimum=0,
                                               outputMaximum=255)

        # Convert from [depth, width, height] to [width, height, depth]
        image.data = sitk.GetArrayFromImage(
            sitk_image).transpose(2, 1, 0)
        image.dims = np.shape(image.data)

        return sitk_image, image
