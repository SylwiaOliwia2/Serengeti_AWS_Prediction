# Serengeti_AWS_Prediction

The repository contains the **code and model to deploy on AWS EC2 to predict** if a set of images from Serengeti Dataset is *blank* (no animals visible) or *non-blank* (animals visible on the picture). It's a proof of concept model, which aim to exclude *blank* images, which bring no information regarding animal behaviour research. In my [blog post](https://sylwiamielnicka.com/blog/image-data-exploration-serengeti-dataset/) you can find more about **data exploration**. I described the process of training model in the [Kaggle notebook](https://www.kaggle.com/sylwiamielnicka/camera-trap-image-identifier-pytorch-cyclicallr).

I hope that this repository might be also helpful as example respository to **Image Recognition with AWS** using **Docker**, **Pytorch** and **Boto3**.

# Prerequests

1. **AWS EC2 instance** (setup instruction here, from [Hosting the docker container on an AWS ec2 instance](https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf#fd93)):
    - Deep Learning AMI (Ubuntu 16.04)
    - p2.xlarge
    - Public DNS automatically set up when the instance is started (in the Step 3 of launching machine choose *Enable* in *Auto-assign Public IP* field)
    - HTTP traffic on port 80
2. AWS EC2 **Key Pair** saved in your computer (also in the [instruction](https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf#fd93)).
3. **S3 Bucket** named *serengeti-images*, containing folder *test_images*. You can place images to predict (from the Serengeti Dataset) inside the folder.


# Deployment

1. Clone the repo:
    ```
    git clone https://github.com/SylwiaOliwia2/Serengeti_AWS_Prediction.git
    ````
2. Start the EC2 instance.
3. SSH to it using Public DNS and saved Permission Key: 
    ```
    ssh -i </home/your-permission-key-location.pem> ubuntu@<your-public-DNS>
    ```
4. In the AWS console:
    ```
    mkdir deploy_folder
    ```
5. Open new console and copy files from your machine to the AWS instance (replace the angle bracket strings with proper values): 
    ```
    scp -i </home/your-permission-key-location.pem> * ubuntu@<your-public-DNS>:/home/ubuntu/deploy_folder 
    scp -i </home/your-permission-key-location.pem> static/* ubuntu@<your-public-DNS>:/home/ubuntu/deploy_folder/static 
    scp -i </home/your-permission-key-location.pem> templates/* ubuntu@<your-public-DNS>:/home/ubuntu/deploy_folder/templates
    ```
    If the Dockerfile will be located directly in `/home/ubuntu` (instead of *deploy_folder*), the Docker will copy all the files in `/home` including preinstalled libraries and environments which is useless and takes ages.
6. In the AWS console: 
    ```
    cd deploy_folder
    ```
7. In the AWS console: 
    ```
    docker build -t app-serengeti . 
    ```
8. In the AWS console: 
    ```
    docker run --gpus all -p 80:5000 app-serengeti . 
    ```
    The console shoud say that th eapp is running on *0.0.0.5000*.
9. In the browser open your public DNS link. In the window you will see simple GUI with the *Predict* button. Press it. The AWS console should display progress in predicting images. Predicting ~40000 images resized to 500x500px took me ~2,5hours.
9. The result will be saved directly in S3 bucket *serengeti-images* as *output_blank_non_test.csv* with the following columns:
    - filenames (csv index)
    - label - predicted label: 0 (blank) or 1 (non-blank)
    - blank_proba - probability if an image being blank
    - non_blank_proba - probability if an image being non_blank


# TODO
**Add safer method of authentication**

According to AWS it's safer to create IAM role and create Temporary Security Credencials for the role, with restricted permissions. I tried to follow [the instruction](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html) but it seems to be outdated.