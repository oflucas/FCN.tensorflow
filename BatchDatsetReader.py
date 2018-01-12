"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_shape = shape of output image - does bilinear resize
        color = True/False
        infer = True/False, the mode to infer a picture from input;
                if True, will have zero annotations everywhere
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()
        self.reset_batch_offset()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        print ('images shape:     ', self.images.shape)
        
        if self.image_options.get('infer', False):
            self.annotations = np.zeros(self.images.shape[:3])
            return
        
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print ('annotations shape:', self.annotations.shape)

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False):
            if 'resize_shape' not in self.image_options:
                h_ = image.shape[0] / 32 * 32 # floor to multiple of 32
                w_ = image.shape[1] / 32 * 32
                self.image_options['resize_shape'] = (h_, w_)
                print 'input image shape preprocess:', image.shape, '->', (h_, w_)

            toShape = self.image_options['resize_shape']
            resize_image = misc.imresize(image,
                                         [toShape[0], toShape[1]], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def next_sequential_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        last_batch = (self.batch_offset >= self.images.shape[0])
        if last_batch:
            self.batch_offset = self.images.shape[0]
        return self.images[start:self.batch_offset], self.annotations[start:self.batch_offset], last_batch
