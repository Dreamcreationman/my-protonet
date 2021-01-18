from PIL import Image
import shutil
import csv
import os
from tqdm import tqdm
import glob
import torch.utils.data as data


class MiniImagenetDataset(data.Dataset):

    processed_folder = "processed/"
    image_folder = "images/"

    def __init__(self, root="../data/miniImagetnet", mode="train", transform=None, target_transform=None):
        """
        Load the data from the original root for dataloader
        :param root:
        :param mode:tran/test/val
        :param transform:
        :param target_transform:
        """
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if not self._check_exist():
            raise RuntimeError("Dataset Not Found Error!")

        self.all_item, self.labels = self.get_all_item()
        self.idx_classes = self.index_classes(self.labels)
        self.y = [self.idx_classes[l] for l in self.labels]

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, idx):
        path, label = self.all_item[idx]
        target = self.idx_classes[label]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def get_all_item(self):
        all_img = []
        labels = []
        with open(os.path.join(self.root, self.mode + '.csv'), 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                path = os.path.join(self.root, self.processed_folder, self.mode, r[1], r[0])
                label = r[1]
                all_img.append((path, label))
                labels.append(label)
        print("== {} Dataset: Found {} items ==".format(self.mode, len(all_img)))
        return all_img, labels

    def _check_exist(self):
        if not os.path.exists(os.path.join(self.root, self.image_folder)):
            return False
        elif not os.path.exists(os.path.join(self.root, self.processed_folder)):
            self.process_original()
        else:
            return True

    def process_original(self):
        all_images = glob.glob(os.path.join(self.root, self.image_folder) + "*" if self.image_folder.endswith(os.sep) else os.path.join(self.root, self.image_folder) + os.sep + "*")
        os.mkdir(os.path.join(self.root, self.processed_folder))
        for image_file in tqdm(all_images, desc="uniform images to same size"):
            im = Image.open(image_file)
            im = im.resize((84, 84), resample=Image.LANCZOS)
            im.save(image_file)

        for datatype in ['train', 'val', 'test']:
            os.mkdir(os.path.join(self.root, self.processed_folder, datatype))
            with open(os.path.join(self.root, datatype + '.csv'), 'r', encoding="utf-8") as f:
                reader = csv.reader(f)
                last_label = ''
                next(reader)
                for row in tqdm(reader, desc="copy processed images"):
                    label = row[1]
                    image_name = row[0]
                    if label != last_label:
                        cur_dir = os.path.join(self.root, self.processed_folder, datatype, label)
                        os.mkdir(cur_dir)
                        last_label = label
                    shutil.copy(os.path.join(self.root, self.image_folder, image_name), cur_dir)

    def index_classes(self, item):
        idx = {}
        for x in item:
            if not x in idx.keys():
                idx[x] = len(idx)
        return idx


if __name__ == '__main__':
    MiniImagenetDataset("/home/data/dataset/mini-imagenet")