import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np


# -------------------------------------------------------------------
# All Data Augmentation Strategies from the original code
# -------------------------------------------------------------------

def cv_random_flip(img, label, depth, edge):
    """
    Applies a random horizontal flip to the image and its corresponding maps.
    """
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth, edge


def randomCrop(image, label, depth, edge):
    """
    Performs a random, centered crop on the image and its maps.
    """
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1, (image_height + crop_win_height) >> 1
    )
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, depth, edge):
    """
    Applies a random rotation to the image and its maps with a 20% probability.
    """
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, depth, edge


def colorEnhance(image):
    """
    Applies random enhancements to brightness, contrast, color, and sharpness.
    """
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    """
    Adds Gaussian noise to a grayscale image.
    """

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    """
    Adds salt-and-pepper noise to a grayscale image.
    """
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# -------------------------------------------------------------------
# Dataset for training
# -------------------------------------------------------------------
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.gts = []
        self.depths = []
        self.edges = []

        # Robust file matching based on filenames
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        label_exts = ['.png', '.jpg', '.jpeg', '.bmp']

        image_files = sorted(os.listdir(image_root))

        print("--- Starting Training Dataset File Matching ---")
        unmatched_count = 0
        for image_name in image_files:
            base_name, ext = os.path.splitext(image_name)
            if ext.lower() not in image_exts:
                continue

            gt_path = self._find_matching_file(gt_root, base_name, label_exts)
            depth_path = self._find_matching_file(depth_root, base_name, label_exts)
            edge_path = self._find_matching_file(edge_root, base_name, label_exts)

            if gt_path and depth_path and edge_path:
                image_path = os.path.join(image_root, image_name)
                # Optional: check if images can be opened and have same size
                try:
                    img_size = Image.open(image_path).size
                    gt_size = Image.open(gt_path).size
                    depth_size = Image.open(depth_path).size
                    edge_size = Image.open(edge_path).size
                    if img_size == gt_size == depth_size == edge_size:
                        self.images.append(image_path)
                        self.gts.append(gt_path)
                        self.depths.append(depth_path)
                        self.edges.append(edge_path)
                    else:
                        print(f"⚠️  Warning: Skipping '{image_name}' due to size mismatch.")
                        unmatched_count += 1
                except Exception as e:
                    print(f"❌ Error: Could not process file for '{base_name}'. Skipping. Error: {e}")
                    unmatched_count += 1
            else:
                unmatched_count += 1

        print("--- Training Dataset File Matching Finished ---")
        print(f"✅ Found {len(self.images)} fully matched training samples.")
        if unmatched_count > 0:
            print(f"ℹ️  Skipped {unmatched_count} files due to missing pairs or errors.")

        assert len(self.images) == len(self.gts) == len(self.depths) == len(self.edges)
        if len(self.images) == 0:
            raise RuntimeError("Error: No valid data found. Please check your dataset paths and filenames.")

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.edges_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def _find_matching_file(self, root_dir, base_name, extensions):
        for ext in extensions:
            file_path = os.path.join(root_dir, base_name + ext)
            if os.path.exists(file_path):
                return file_path
        return None

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        edge = self.binary_loader(self.edges[index])

        # Apply data augmentations on PIL images
        image, gt, depth, edge = cv_random_flip(image, gt, depth, edge)
        image, gt, depth, edge = randomCrop(image, gt, depth, edge)
        image, gt, depth, edge = randomRotation(image, gt, depth, edge)
        image = colorEnhance(image)
        # gt = randomGaussian(gt) # This was commented out in the original, kept here for reference
        gt = randomPeper(gt)

        # Transform PIL images to Tensors
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        edge = self.edges_transform(edge)

        return image, gt, depth, edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# -------------------------------------------------------------------
# Dataloader for training
# -------------------------------------------------------------------
def get_loader(image_root, gt_root, depth_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=16,
               pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, depth_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# -------------------------------------------------------------------
# Test dataset and loader
# -------------------------------------------------------------------
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = []
        self.gts = []
        self.depths = []
        self.names = []

        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        label_exts = ['.png', '.jpg', '.jpeg', '.bmp']

        image_files = sorted(os.listdir(image_root))

        print("\n--- Initializing Test Dataset ---")
        for image_name in image_files:
            base_name, ext = os.path.splitext(image_name)
            if ext.lower() not in image_exts:
                continue

            gt_path = self._find_matching_file(gt_root, base_name, label_exts)
            depth_path = self._find_matching_file(depth_root, base_name, label_exts)

            if gt_path and depth_path:
                self.images.append(os.path.join(image_root, image_name))
                self.gts.append(gt_path)
                self.depths.append(depth_path)
                self.names.append(base_name)
            else:
                print(f"⚠️  Warning (Test Set): Skipping '{image_name}' due to missing pairs.")

        print(f"✅ Found {len(self.images)} fully matched test samples.")
        if len(self.images) == 0:
            raise RuntimeError("Error: No valid test data found.")

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def _find_matching_file(self, root_dir, base_name, extensions):
        for ext in extensions:
            file_path = os.path.join(root_dir, base_name + ext)
            if os.path.exists(file_path):
                return file_path
        return None

    def load_data(self):
        if self.index >= self.size:
            # Reset index to loop through the dataset again if needed
            self.index = 0
            return None, None, None, None, None

        image_path = self.images[self.index]
        gt_path = self.gts[self.index]
        depth_path = self.depths[self.index]

        image = self.rgb_loader(image_path)
        gt = self.binary_loader(gt_path)

        # Keep original image for post-processing or size reference
        original_image_for_post = image.copy()
        original_image_for_post = original_image_for_post.resize(gt.size)

        image_tensor = self.transform(image).unsqueeze(0)

        depth = self.binary_loader(depth_path)
        depth_tensor = self.depths_transform(depth).unsqueeze(0)

        name = self.names[self.index]

        # The output name should consistently be .png for evaluation
        output_name = name + '.png'

        self.index += 1
        return image_tensor, gt, depth_tensor, output_name, np.array(original_image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size