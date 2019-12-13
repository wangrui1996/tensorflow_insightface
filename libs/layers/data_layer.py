import os
import numpy as np
from keras_preprocessing import image
from tensorflow.python.keras import backend
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

class DirectoryIterator(image.DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = image.load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = image.img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = image.array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return {"inputs": batch_x, "labels": batch_y}, batch_y
        else:
            return {"inputs": batch_x, "labels" : batch_y}, batch_y, self.sample_weight[index_array]

@keras_export('keras.image.DataGenerator')
class ImageDataGenerator(image.ImageDataGenerator):
  """Generate batches of tensor image data with real-time data augmentation.

   The data will be looped over (in batches).

  Arguments:
      featurewise_center: Boolean.
          Set input mean to 0 over the dataset, feature-wise.
      samplewise_center: Boolean. Set each sample mean to 0.
      featurewise_std_normalization: Boolean.
          Divide inputs by std of the dataset, feature-wise.
      samplewise_std_normalization: Boolean. Divide each input by its std.
      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
      zca_whitening: Boolean. Apply ZCA whitening.
      rotation_range: Int. Degree range for random rotations.
      width_shift_range: Float, 1-D array-like or int
          - float: fraction of total width, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-width_shift_range, +width_shift_range)`
          - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `width_shift_range=[-1, 0, +1]`,
              while with `width_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      height_shift_range: Float, 1-D array-like or int
          - float: fraction of total height, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-height_shift_range, +height_shift_range)`
          - With `height_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `height_shift_range=[-1, 0, +1]`,
              while with `height_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      brightness_range: Tuple or list of two floats. Range for picking
          a brightness shift value from.
      shear_range: Float. Shear Intensity
          (Shear angle in counter-clockwise direction in degrees)
      zoom_range: Float or [lower, upper]. Range for random zoom.
          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
      channel_shift_range: Float. Range for random channel shifts.
      fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
          Default is 'nearest'.
          Points outside the boundaries of the input are filled
          according to the given mode:
          - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          - 'nearest':  aaaaaaaa|abcd|dddddddd
          - 'reflect':  abcddcba|abcd|dcbaabcd
          - 'wrap':  abcdabcd|abcd|abcdabcd
      cval: Float or Int.
          Value used for points outside the boundaries
          when `fill_mode = "constant"`.
      horizontal_flip: Boolean. Randomly flip inputs horizontally.
      vertical_flip: Boolean. Randomly flip inputs vertically.
      rescale: rescaling factor. Defaults to None.
          If None or 0, no rescaling is applied,
          otherwise we multiply the data by the value provided
          (after applying all other transformations).
      preprocessing_function: function that will be implied on each input.
          The function will run after the image is resized and augmented.
          The function should take one argument:
          one image (Numpy tensor with rank 3),
          and should output a Numpy tensor with the same shape.
      data_format: Image data format,
          either "channels_first" or "channels_last".
          "channels_last" mode means that the images should have shape
          `(samples, height, width, channels)`,
          "channels_first" mode means that the images should have shape
          `(samples, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
      dtype: Dtype to use for the generated arrays.

  Examples:

  Example of using `.flow(x, y)`:

  ```python
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  datagen = ImageDataGenerator(
      featurewise_center=True,
      featurewise_std_normalization=True,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True)
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(x_train)
  # fits the model on batches with real-time data augmentation:
  model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                      steps_per_epoch=len(x_train) / 32, epochs=epochs)
  # here's a more "manual" example
  for e in range(epochs):
      print('Epoch', e)
      batches = 0
      for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
          model.fit(x_batch, y_batch)
          batches += 1
          if batches >= len(x_train) / 32:
              # we need to break the loop by hand because
              # the generator loops indefinitely
              break
  ```

  Example of using `.flow_from_directory(directory)`:

  ```python
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1./255)
  train_generator = train_datagen.flow_from_directory(
          'data/train',
          target_size=(150, 150),
          batch_size=32,
          class_mode='binary')
  validation_generator = test_datagen.flow_from_directory(
          'data/validation',
          target_size=(150, 150),
          batch_size=32,
          class_mode='binary')
  model.fit_generator(
          train_generator,
          steps_per_epoch=2000,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=800)
  ```

  Example of transforming images and masks together.

  ```python
  # we create two instances with the same arguments
  data_gen_args = dict(featurewise_center=True,
                       featurewise_std_normalization=True,
                       rotation_range=90,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       zoom_range=0.2)
  image_datagen = ImageDataGenerator(**data_gen_args)
  mask_datagen = ImageDataGenerator(**data_gen_args)
  # Provide the same seed and keyword arguments to the fit and flow methods
  seed = 1
  image_datagen.fit(images, augment=True, seed=seed)
  mask_datagen.fit(masks, augment=True, seed=seed)
  image_generator = image_datagen.flow_from_directory(
      'data/images',
      class_mode=None,
      seed=seed)
  mask_generator = mask_datagen.flow_from_directory(
      'data/masks',
      class_mode=None,
      seed=seed)
  # combine generators into one which yields image and masks
  train_generator = zip(image_generator, mask_generator)
  model.fit_generator(
      train_generator,
      steps_per_epoch=2000,
      epochs=50)
  ```
  """

  def __init__(self,
               featurewise_center=False,
               samplewise_center=False,
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               zca_epsilon=1e-6,
               rotation_range=0,
               width_shift_range=0.,
               height_shift_range=0.,
               brightness_range=None,
               shear_range=0.,
               zoom_range=0.,
               channel_shift_range=0.,
               fill_mode='nearest',
               cval=0.,
               horizontal_flip=False,
               vertical_flip=False,
               rescale=None,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0,
               dtype=None):
    if data_format is None:
      data_format = backend.image_data_format()
    kwargs = {}
    if 'dtype' in tf_inspect.getfullargspec(
        image.ImageDataGenerator.__init__)[0]:
      if dtype is None:
        dtype = backend.floatx()
      kwargs['dtype'] = dtype
    super(ImageDataGenerator, self).__init__(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split,
        **kwargs)

  def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        """Takes the path to a directory & generates batches of augmented data.

        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.

        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )



