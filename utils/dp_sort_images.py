import shutil
import os
import pandas as pd

class SORT_DEDUPED:
    def __init__(self, dataset_path):
        # Set the directories of the source images, target sorted images, and target test images
        self.dataset_dir = dataset_path
        self.raw_img_dir = self.dataset_dir + '/deduped_images'
        self.sorted_img_dir = self.dataset_dir + '/sorted_images'
        self.test_img_dir = self.dataset_dir + '/sorted_test'
    
    def get_duplicates(self):
        # Get the duplicate images
        dup_list = open("./dataset/duplicates.txt").read().splitlines()
        return dup_list

    def remove_dirs(self):
        # Remove the sorted_images and the sorted_test
        if os.path.exists(self.sorted_img_dir):
            shutil.rmtree(self.sorted_img_dir)
        
        if os.path.exists(self.test_img_dir):
            shutil.rmtree(self.test_img_dir)

    def get_sort_info(self, set = None):
        if set == 'train':
            ## TRAINING IMAGES
            # Get the location information to be used for setting the sorted images
            train_info = pd.read_csv(self.dataset_dir + '/train.csv', dtype = 'string')
            train_info['filename'] = train_info['Id'].astype(str) + '.jpg'

            sorted_img_loc = dict(zip(train_info['filename'], train_info['label']))

            return sorted_img_loc
        elif set == 'test':
            ## TEST IMAGES
            # Get the location information to be used for setting the test images
            test_info = pd.read_csv(self.dataset_dir + '/test.csv', dtype = 'string')
            test_sort_info = test_info['Id'].tolist()
            test_sort_info = [s + '.jpg' for s in test_sort_info]

            return test_sort_info

    def main(self):
        dup_list = self.get_duplicates()

        self.remove_dirs()

        train_sort_info = self.get_sort_info(set='train')
        test_sort_info = self.get_sort_info(set='test')

        # Move all the training images
        for file, label in train_sort_info.items():
            if file in dup_list:
                pass
            else:
                src = self.raw_img_dir+ '/' + file
                dst = self.sorted_img_dir  + '/' + label + '/' + file
                
                if os.path.exists(self.sorted_img_dir   + '/' + label):
                        shutil.copy(src, dst)
                else:
                    os.makedirs(self.sorted_img_dir + '/' + label)
                    shutil.copy(src, dst)

        # Move all the test images
        for file in test_sort_info:
            if file in dup_list:
                pass
            else:
                src = self.raw_img_dir+ '/' + file
                dst = self.test_img_dir + '/' + file
                
                if os.path.exists(self.test_img_dir):
                    shutil.copy(src, dst)
                else:
                    os.makedirs(self.test_img_dir)
                    shutil.copy(src, dst)

def call_class(dataset_path):
    a = SORT_DEDUPED(dataset_path)
    a.main()