import os
import boto3
import logging
from io import BytesIO
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad as TorchPad
ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)


class CreateDataset(Dataset):
    def __init__(self, bucketname, file_df, transform):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucketname)

        filepaths = []
        for file in bucket.objects.all():
            if file.key.startswith("test-images"):  # docelowo folder "images"
                filepaths.append(file.key)
        self.image_filepaths_list = [f for f in filepaths if f.split("/")[-1] in file_df.index]
        filenames = [f.split("/")[-1] for f in self.image_filepaths_list]
        self.label_files_df = file_df.loc[filenames]

        self.bucketname = bucketname
        self.transform = transform

    def __len__(self):
        return len(self.image_filepaths_list)

    def __getitem__(self, idx):
        filepath = self.image_filepaths_list[idx]
        filename = filepath.split("/")[-1]
        s3 = boto3.client('s3')
        file_byte_string = s3.get_object(Bucket=self.bucketname,
                                         Key=filepath)['Body'].read()
        try:
            image = Image.open(BytesIO(file_byte_string))
            image = self.transform(image)
        except:
            file_byte_string = s3.get_object(Bucket=self.bucketname,
                                             Key="broken.jpg")['Body'].read()
            image = Image.open(BytesIO(file_byte_string))
            filename = "broken_" + filename
            log.info("################ %s corrupted " % filename)
        # true labels are not needed'as we predict based on the model, don't validate the prediction in this process
        # label = self.label_files_df.loc[filename, "labels"]
        return image, filename


class Resize2(object):
    '''
    Resize object along LONGER border (tarnsforms.Resize works along smaller border)
    '''

    def __init__(self, new_max_size, interpolation=Image.NEAREST):
        self.new_max_size = new_max_size
        self.interpolation = interpolation

    def __call__(self, img):
        old_size = img.size[:2]
        ratio = float(self.new_max_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        return img.resize(new_size, resample=self.interpolation)


class SquarePad(object):
    '''
    Square img by extending smaller border to longer border size and filling empty space, so that both sides will have the same size
    '''

    def __init__(self, sqr_size, padding_mode="reflect"):
        self.sqr_size = sqr_size
        self.padding_mode = padding_mode

    def __call__(self, img):
        old_size = img.size[:2]
        pad_size = [0, 0]
        pad_size.extend([self.sqr_size - x for x in old_size])
        return TorchPad(img, tuple(pad_size), padding_mode=self.padding_mode)
