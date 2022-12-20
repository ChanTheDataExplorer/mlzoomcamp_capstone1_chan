import utils.dp_dedup as Dedup
import utils.dp_sort_images as SORT_DEDUPED
import utils.dp_split_train_val as SplitTV

import os, shutil

class DataPrep:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.raw_img_dir = self.dataset_path + '/images'
        self.deduped_img_dir = self.dataset_path + '/deduped_images'
        self.sorted_img_dir = self.dataset_path + '/sorted_images'
        self.sorted_img_train = self.dataset_path + '/sorted_images_train'
        self.sorted_img_val = self.dataset_path + '/sorted_images_val'
        self.sorted_img_test = self.dataset_path + '/sorted_test'

    def remove_dirs(self, phase = None):
        if phase == 'pre':
            retain_folders = ['images']
            retain_files = ['train.csv','test.csv','sample_submission.csv']

        if phase == 'post':
            retain_folders = ['images', 'sorted_images_train', 'sorted_images_val', 'sorted_test']
            retain_files = ['train.csv','test.csv','sample_submission.csv']
        
        retain_list = retain_folders + retain_files
        for file in os.listdir(self.dataset_path):
            if file not in retain_list:
                try:
                    os.remove(self.dataset_path + '/' + file)
                except:
                    shutil.rmtree(self.dataset_path + '/' + file)

    def main(self):
        self.remove_dirs(phase = 'pre')
        
        Dedup.call_class(self.dataset_path)
        SORT_DEDUPED.call_class(self.dataset_path)
        SplitTV.call_class(self.dataset_path)

        self.remove_dirs(phase = 'post')

        print('Successful data preparation')

def call_class(dataset_path = '/dataset'):
    a = DataPrep(dataset_path)
    a.main()