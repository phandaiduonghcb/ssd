"""face_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import csv
import os
import xml.etree.ElementTree as ET
import glob
import numpy as np

# TODO(face_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(face_dataset): BibTeX citation
_CITATION = """
"""
CLASS_DICT = {
  'face' : 0,
}
DATASET_PATH = '/dl/ssd/datasets/face_data'
NUM_TRAIN_EXAMPLES = len(glob.glob(os.path.join(DATASET_PATH, 'train/*.xml')))
NUM_VAL_EXAMPLES = len(glob.glob(os.path.join(DATASET_PATH, 'valid/*.xml')))
NUM_TEST_EXAMPLES = len(glob.glob(os.path.join(DATASET_PATH, 'test/*.xml')))

class FaceDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for face_dataset dataset."""
  path = DATASET_PATH
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
            'size': tfds.features.Tensor(shape=(2,), dtype=tf.int32),
            'image_filename': tfds.features.Text(),
            'objects': tfds.features.Tensor(shape=(None, 5), dtype=tf.float32)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    # path = '/dl/ssd/datasets/Vehicles-OpenImages.v1i.voc'

    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    self.num_train_examples = len(glob.glob(os.path.join(self.path, 'train/*.xml')))
    self.num_valid_examples = len(glob.glob(os.path.join(self.path, 'valid/*.xml')))
    self.num_test_examples = len(glob.glob(os.path.join(self.path, 'test/*.xml')))
    return {
        'train': self._generate_examples(
          images_path=os.path.join(self.path, 'train')),
        'valid': self._generate_examples(
          images_path=os.path.join(self.path, 'valid')),
        'test': self._generate_examples(
          images_path=os.path.join(self.path, 'test')),
    }
  def _extract_xml(self, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    size = root.find('size')
    size = (int(size[1].text), int(size[0].text))
    for member in root.findall('object'):
      # print(member.keys())
      # return
      w, h = int(root.find('size')[0].text), int(root.find('size')[1].text)
      cls_name = member[0].text
      object = (
                CLASS_DICT[cls_name],
                int(member[4][0].text)/w,
                int(member[4][1].text)/h,
                int(member[4][2].text)/w,
                int(member[4][3].text)/h
                )
      if (object[1] > object[3] or object[2] > object[4]):
        raise AssertionError(f'False bounding box: {xml_path}\n {object}')
      objects.append(object)
    return objects, size


  def _generate_examples(self, images_path):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the 
    image_paths = glob.glob(os.path.join(images_path, '*.jpg'))
    for image_path in image_paths:
      # And yield (key, feature_dict)
      image_id = image_path.split('/')[-1]
      xml_filename = os.path.splitext(image_id)[0] + '.xml'
      objects, size = self._extract_xml(os.path.join(images_path, xml_filename))
      if objects:
        yield image_id, {
            'image': os.path.join(images_path , image_id),
            'image_filename': image_id,
            'objects': objects,
            'size':size,
        }
      else:
        print(f"WARNING: Skip {xml_filename} because len(objects) = 0")
        continue
