import numpy as np
import pandas as pd
from classes import CreateDataset, Resize2, SquarePad
from func import model_predict, random_seed
import torch
from torchvision import transforms
import boto3
from botocore.exceptions import ClientError
from io import StringIO
import logging


log = logging.getLogger(__name__)


def predict_files():
    ###############################################################
    ## INIT DATA
    classes = ['blank','non_blank']

    MAX_SIZE = 500
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
            Resize2(MAX_SIZE),
            SquarePad(MAX_SIZE),
            transforms.ColorJitter(brightness=(0.9, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(color_mean, color_std)
        ])

    # BATCH SIZE is applied in case you cannot fit all the data in the CPU/GPU RAM at the same time and batch processing is needed.
    # SOURCE: https://stackoverflow.com/a/37912576
    BATCH_SIZE = 100

    # labels only in case I'll come out to validate prediction in this process
    labels = pd.read_csv("labels_blank_non_blank.csv", index_col=0).fillna(0)  # it's inside the docker
    bucket_images_folder = "serengeti-images"

    log.info("################ Creating dataset.")
    dataset = CreateDataset(bucket_images_folder, labels, transform)
    log.info("################ dataset length: %d" % len(dataset))
    log.info("################ First three filepaths:")
    log.info(dataset.image_filepaths_list[:3])

    random_seed(123)
    log.info("################ Loading data")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    log.info("################ data_loader length: %d" % len(data_loader))
    device = torch.device("cuda")


    ###############################################################
    ## LOAD MODEL AND PREDICT
    ###############################################################
    model = torch.load("model_blank_non.pkl", map_location=device)
    model.eval()

    log.info("################ START PREDICTING:")
    pred, filenames, all_proba = model_predict(model, data_loader, device, log)
    log.info("### all_proba: ")
    log.info(all_proba)


    ###############################################################
    ## UPLOAD RESULTS TO S3
    ###############################################################
    log.info("################ UPLOAD RESULTS TO S3:")
    data = np.concatenate([pred.reshape(-1, 1).astype(int), all_proba], axis=1)
    cols = ["pred_label"] + [x + "_proba" for x in classes]
    df = pd.DataFrame(data, columns=cols, index=filenames)
    log.info("################ output_df")
    log.info(df.head())

    s3_client = boto3.client('s3')

    try:
        csv_buf = StringIO()
        df.to_csv(csv_buf, header=True, index=True)
        csv_buf.seek(0)
        s3_client.put_object(ACL='private',
                             Body=csv_buf.getvalue(),
                             Bucket='serengeti-images',
                             Key='output_blank_non_test.csv')
        csv_buf.close()
    except ClientError as e:
        logging.error(e)
    log.info("################ DONE")
