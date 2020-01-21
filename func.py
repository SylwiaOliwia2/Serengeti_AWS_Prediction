import random
import torch
import torch.nn.functional as F
import numpy as np


def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def model_predict(model, data_loader, device, log):
    '''
    Predeict image labels for all the batches
    :param model: torch model
    :param data_loader: ttorch.utils.data.DataLoader
    :param device: torch.device
    :param log: logging.getLogger variable
    :return:
    '''
    all_pred = np.array([])
    filenames = []
    all_proba = []

    with torch.no_grad():
        for batch_nr, data in enumerate(data_loader):
            log.info("################ Predicting batch %d out of %d" % (batch_nr, len(data_loader)))
            X, filename = data
            X= X.to(device)
            out = model(X)
            # about softmax - https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/6?u=mielnicka
            out = F.softmax(out, dim=1)
            for p in out:
                all_proba.append(p.to("cpu").tolist())
            _, pred = torch.max(out, 1)
            # append values to arrays
            all_pred = np.append(all_pred, pred.to("cpu"))
            filenames.extend(list(filename))
    return all_pred, filenames, np.array(all_proba)
