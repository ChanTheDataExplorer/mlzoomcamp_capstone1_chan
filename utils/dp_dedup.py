import os
import shutil

from collections import defaultdict
from hashlib import md5
from pathlib import Path

import PIL
            
class Dedup:
    def __init__(self, dataset_path):
        # Set the directories of the source images, target sorted images, and target test images
        self.dataset_dir = dataset_path
        self.raw_img_dir = self.dataset_dir + '/images'
        self.deduped_img_dir = self.dataset_dir + '/deduped_images'

    def get_duplicates(self):
        # CHECKING OF DUPLICATES
        # IF FOUND, SAVE TO duplicate_all and for_removal as a list
        image_dir = Path(self.raw_img_dir)

        hash_dict = defaultdict(list)
        for image in image_dir.glob('*.jpg'):
            with image.open('rb') as f:
                img_hash = md5(f.read()).hexdigest()
                hash_dict[img_hash].append(image)

        duplicate_all = []
        for_removal = []
        for k, v in hash_dict.items():
            duplicate_pair = []

            if len(v) > 1:
                if v[0].name != v[1].name:
                    duplicate_pair.append(v[0])
                    duplicate_pair.append(v[1])

                    for_removal.append(v[1])
            
                duplicate_all.append(duplicate_pair)

        # Change the data type from Posixpath to String
        for_removal = list(map(str, for_removal))

        return for_removal
        
    # Remove file function
    @staticmethod
    def remove_file(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)

    def main(self):
        raw_img_dir = self.raw_img_dir
        deduped_img_dir = self.deduped_img_dir

        for_removal = self.get_duplicates()
         
        # Save list of for removal images in a txt file
        with open('./dataset/duplicates.txt', 'w') as f:
            for image in for_removal:
                name = os.path.basename(image)
                f.write(f"{name}\n")

        # Copy all the files from the downloaded images to a new directory
        if not os.path.exists(deduped_img_dir):
            shutil.copytree(raw_img_dir, deduped_img_dir)
        else:
            shutil.rmtree(deduped_img_dir)
            shutil.copytree(raw_img_dir, deduped_img_dir)

        # Remove all the items in for_removal list from the new directory
        for dup in for_removal:
            name = os.path.basename(dup)
            self.remove_file(name, deduped_img_dir)

            print(f'File {name} is a duplicate image. File is removed')

def call_class(dataset_path):
    a = Dedup(dataset_path)
    a.main()