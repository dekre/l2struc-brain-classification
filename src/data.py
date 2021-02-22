import numpy as np
import zipfile as zip_
from pydicom import dcmread
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self, 
            files: np.array, 
            labels: np.array,            
            src_zip_path: str,    
            brain_slice_pos: tuple = (3,3),
            brain_segments: tuple = (8,8),
            batch_size=32, 
            dim=(128,128), 
            n_channels=1,
            n_classes=2, 
            shuffle=True
        ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.files = files
        self.brain_slice_pos = brain_slice_pos
        self.brain_segments = brain_segments
        self.src_zip = zip_.ZipFile(src_zip_path)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_ = [(self.files[k], self.labels[k],) for k in indexes]        

        # Generate data
        X, y = self.__data_generation(data_)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, data: list):   
        """Generates data containing batch_size samples
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)     
        
        for idx, (file, label) in enumerate(data):
            source_pixel_array = self.__get_brain_image_pixel_array(self.src_zip, file)
            target_pixel_array = self.__get_brain_slice(source_pixel_array, self.brain_slice_pos, self.brain_segments)
            X[idx, ] = np.expand_dims(target_pixel_array, axis=-1)
            y[idx] = label

        return X, y        


    def __get_brain_slice(self, pixel_array: np.ndarray, pos: tuple = (0,0), segments: tuple = (8,8)):    
        total_pxiel = np.array(pixel_array.shape)
        pos, segments = np.array(pos), np.array(segments)
        rng_start = (total_pxiel / segments * pos).astype(int)
        rng_end = (rng_start + total_pxiel / segments).astype(int)    
        return pixel_array[rng_start[0]:rng_end[0],rng_start[1]:rng_end[1]]


    def __get_brain_image_pixel_array(self, src_zip: zip_.ZipFile, img_path: str):
        with src_zip.open(img_path, "r") as f:
            ds = dcmread(f)
            f.close()    
        return ds.pixel_array