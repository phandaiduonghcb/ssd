import tensorflow as tf

class BatchDatasetForOD():
    def __init__(self, tfds_dataset, batch_size, image_size):
        self.tfds_dataset = tfds_dataset
        self.batch_size = batch_size
        self.image_size = image_size

    def __iter__(self):
        self.iterator = iter(self.tfds_dataset)
        self.end = False
        return self

    def __next__(self):
        X = []
        Y = []
        if self.end:
            raise StopIteration

        while len(X) != self.batch_size:
            try:
                current_example = next(self.iterator)
            except StopIteration:
                self.end=True
                break
            resized_image = tf.image.resize(current_example['image'], self.image_size)
            X.append(resized_image)
            Y.append(current_example['objects'])

        return tf.convert_to_tensor(X), Y