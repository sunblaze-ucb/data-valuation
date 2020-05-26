import numpy as np
import os
import torch
from sklearn.utils import shuffle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import *
# Deep Feature
from models.resnetF import
from plot_resnet import *


device = torch.device('cuda')
model = resnet18(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d(1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



result_path = "embedding_result/ds/"
tiny_path = "embedding_data/tinyimagenet/resnet18_"
data_path = "embedding_data/ds/"

LOAD_ORIGINAL_DATA = False
CAL_DEEP_FEATURES = False

print("Load original data or not:")
if LOAD_ORIGINAL_DATA == True:
    print("Load original data")
    # load raw data
    data_dir = "data/tiny-imagenet-200"
    data_transforms = transforms.Compose([transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    batch_size = len(image_datasets) # 1000 for test

    # change dataloaders to numpy
    train_dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=64)
    raw_train_X, raw_train_Y = next(iter(train_dataloader))
    raw_train_X = raw_train_X.numpy() # (100000, 3, 64, 64)
    raw_train_Y = raw_train_Y.numpy() # (100000, )
    print("raw data shape: ", raw_train_X.shape, raw_train_Y.shape)

    # load embedding train data
    for i in range(100): # 100
        x = np.load(tiny_path+str(i)+".npz")["x"]
        y = np.load(tiny_path+str(i)+".npz")["y"]
        if i == 0:
            embed_train_X = x
            embed_train_Y = y
        else:
            embed_train_X = np.concatenate((embed_train_X, x), axis=0)
            embed_train_Y = np.concatenate((embed_train_Y, y), axis=0)
    print("train data size: ", embed_train_X.shape, embed_train_Y.shape) # (100000, 512, 1, 1) (100000,)

    embed_train_X, embed_train_Y, raw_train_X, raw_train_Y = shuffle(embed_train_X, embed_train_Y, raw_train_X, raw_train_Y, random_state=0) # shuffle array

    # save data
    for i in range(10):
        np.savez_compressed(data_path + "train/" + "train_" + str(i) + ".npz", x=embed_train_X[i * 10000:i * 10000 + 10000], y=embed_train_Y[i * 10000:i * 10000 + 10000])
        np.savez_compressed(data_path + "raw_train/" + "raw_train_" + str(i) + ".npz", x=raw_train_X[i * 10000:i * 10000 + 10000], y=raw_train_Y[i * 10000:i * 10000 + 10000])
    print("===Data saved===")

if LOAD_ORIGINAL_DATA == False:
    # load embedding data
    print("Load reserved data.")
    for i in range(10):
        with open(data_path + 'train/train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                embed_train_X = x
                embed_train_Y = y
            else:
                embed_train_X = np.concatenate((embed_train_X, x), axis=0)
                embed_train_Y = np.concatenate((embed_train_Y, y), axis=0)
        with open(data_path + 'raw_train/raw_train_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                raw_train_X = x
                raw_train_Y = y
            else:
                raw_train_X = np.concatenate((raw_train_X, x), axis=0)
                raw_train_Y = np.concatenate((raw_train_Y, y), axis=0)


EMBED_SV = True
RAW_SV = False
DEEP_SV = False
# calculate shapley values
k = 6
train_num = 95000  # 85000
cal_num = 5000
print("neighbour number:", k, " train_num:", train_num)
print("train shape: ", embed_train_X[:train_num].shape, "calculate sv shape: ", embed_train_X[train_num:train_num+cal_num].shape)

if EMBED_SV == True:
    embed_knn_values, *_ = old_knn_shapley(k, embed_train_X[:train_num], embed_train_X[train_num:train_num + cal_num], embed_train_Y[:train_num], embed_train_Y[train_num:train_num + cal_num])
    np.savez_compressed(result_path + '_embed_knn.npz', knn=embed_knn_values)

    old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, embed_train_X[:train_num], embed_train_X[train_num:train_num+cal_num], embed_train_Y[:train_num], embed_train_Y[train_num:train_num+cal_num])
    print("loo knn score on embedding data: ", fc1_scores)
    np.savez_compressed(result_path + '_embed_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)

if RAW_SV == True:
    knn_values, *_ = old_knn_shapley(k, raw_train_X[:train_num], raw_train_X[train_num:train_num + cal_num], raw_train_Y[:train_num], raw_train_Y[train_num:train_num + cal_num])
    np.savez_compressed(result_path + 'raw_knn.npz', knn=knn_values)

    old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, raw_train_X[:train_num], raw_train_X[train_num:train_num+cal_num], raw_train_Y[:train_num], raw_train_Y[train_num:train_num+cal_num])
    print("loo knn score on embedding data: ", fc1_scores)
    np.savez_compressed(result_path + 'raw_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)

if CAL_DEEP_FEATURES == True:
    raw_train_X = torch.from_numpy(raw_train_X).contiguous().view(-1, 3,64,64)
    raw_train_Y = torch.from_numpy(raw_train_Y).view(-1,).long()
    batch_size = 128
    epochs = 15
    model = model.to(device)
    plot_tiny_train("train", model, device, raw_train_X[:train_num], raw_train_Y[:train_num], optimizer, criterion, scheduler, batch_size, epochs)
    acc, _ = plot_tiny_train("val", model, device, raw_train_X[:train_num], raw_train_Y[:train_num], optimizer, criterion, scheduler, batch_size, epochs=1)
    print("Model for deep features acc: ", acc)

    deep_features = []
    for inputs, labels in batch(raw_train_X, raw_train_Y, batch_size):
        inputs = inputs.to(device)
        labels = labels.to(device)
        deep_feature, _ = model(inputs)
        deep_features.append(deep_feature.view(deep_feature.size(0), -1).cpu().detach().numpy())
    deep_features_X = np.concatenate(deep_features)
    deep_features_Y = raw_train_Y
    print("deep features shape: ", deep_features.shape)
    # save deep features
    for i in range(10):
        np.savez_compressed(data_path + "deep_features/" + "df_" + str(i) + ".npz", x=deep_features_X[i * 10000:i * 10000 + 10000], y=raw_train_Y[i * 10000:i * 10000 + 10000])
else:
    print("Load deep features data.")
    for i in range(10):
        with open(data_path + 'deep_features/df_' + str(i) + '.npz', "rb") as f:
            x = np.load(f)["x"]
            y = np.load(f)["y"]
            if i == 0:
                deep_features_X = x
                deep_features_Y = y
            else:
                deep_features_X = np.concatenate((deep_features_X, x), axis=0)
                deep_features_Y = np.concatenate((deep_features_Y, y), axis=0)

if DEEP_SV == True:
    deep_knn_values, *_ = old_knn_shapley(k, deep_features_X[:train_num], deep_features_X[train_num:train_num + cal_num], deep_features_Y[:train_num], deep_features_Y[train_num:train_num + cal_num])
    np.savez_compressed(result_path + 'deep_features_knn.npz', knn=deep_knn_values)

    old_fc1_knn_values, fc1_scores, fc1_false = loo_knn_shapley(k, deep_features_X[:train_num], deep_features_X[train_num:train_num+cal_num], deep_features_Y[:train_num], deep_features_Y[train_num:train_num+cal_num])
    print("deep features loo knn score on embedding data: ", fc1_scores)
    np.savez_compressed(result_path + 'deep_features_loo_knn.npz', loo_knn=old_fc1_knn_values, score=fc1_scores, false=fc1_false)
