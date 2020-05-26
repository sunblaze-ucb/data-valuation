import numpy as np
import tensorflow as tf
from plot_resnet import *
# import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

result_path = "embedding_result/da/"

# load mnist embeddings
mnist_embed_path = "embedding_data/da/customMnist/"
with open(mnist_embed_path + "train/resnet18_0.npz", 'rb') as f:
    embed_train_X = np.load(f)['x']
    embed_train_Y = np.load(f)['y']
with open(mnist_embed_path + "test/resnet18_0_test.npz", 'rb') as f:
    embed_cal_sv_X = np.load(f)['x']
    embed_cal_sv_Y = np.load(f)['y']
with open(mnist_embed_path + "prediction/resnet18_0.npz", 'rb') as f:
    embed_x_pre = np.load(f)['x']
    embed_y_pre = np.load(f)['y']
with open(mnist_embed_path + "prediction/resnet18_1.npz", 'rb') as f:
    embed_x_pre = np.concatenate((embed_x_pre, np.load(f)['x']), axis=0)
    embed_y_pre = np.concatenate((embed_y_pre, np.load(f)['y']), axis=0)
print("embed train shape: ", embed_train_X.shape, "embed cal sv shape: ", embed_cal_sv_X.shape, "embed prediction shape: ", embed_x_pre.shape)

# load mnist data
with open(mnist_embed_path + "train/train.npz", 'rb') as f:
    train_X = np.load(f)['x']
    train_Y = np.load(f)['y']
with open(mnist_embed_path + "test/test.npz", 'rb') as f:
    cal_sv_X = np.load(f)['x']
    cal_sv_Y = np.load(f)['y']
with open(mnist_embed_path + "prediction/prediction.npz", 'rb') as f:
    x_pre = np.load(f)['x']
    y_pre = np.load(f)['y']

print("train shape: ", train_X.shape, "cal sv shape: ", cal_sv_X.shape, "prediction shape: ", x_pre.shape)


noise_size = 50
train_size = 100
cal_sv_size = 100
pre_size = 2000
pre_noise_size = 1500
heldout_size = 1000
k = 5

# load mnist raw data with noise
mnist = tf.keras.datasets.mnist
(mnist_train_X, mnist_train_Y), (mnist_test_X, mnist_test_Y) = mnist.load_data()

# mnist_test_X = np.reshape(mnist_test_X, [-1, 28, 28, 1])
mnist_test_X = mnist_test_X.astype(np.float32) / 255
val_X = np.reshape(mnist_test_X[-heldout_size:], (heldout_size, -1))
val_Y = np.reshape(mnist_test_Y[-heldout_size:], (heldout_size, -1))


LOAD_SV = True
if LOAD_SV == False:
    embed_knn_sv, *_ = old_knn_shapley(k, embed_train_X, embed_cal_sv_X, embed_train_Y, embed_cal_sv_Y)
    np.savez_compressed(result_path + 'mnist_embed_knn_' + str(k) + '.npz', knn=embed_knn_sv)

else:
    with open(result_path + 'mnist_embed_knn_' + str(k) + '.npz', 'rb') as f:
        embed_knn_sv = np.load(f)["knn"]
print("mnist embed knn sv shape: ", embed_knn_sv.shape)


# train random forest to predict values
embed_knn_sv = embed_knn_sv / np.linalg.norm(embed_knn_sv)
filted_embed_knn_sv_idxs = np.where(embed_knn_sv >= 0.0)[0]


random_forest =  RandomForestRegressor(max_depth=100, n_estimators=50, random_state=666)
random_forest.fit(embed_train_X[filted_embed_knn_sv_idxs][:,:,0,0], embed_knn_sv[filted_embed_knn_sv_idxs])
embed_knn_pre_scores = random_forest.predict(embed_x_pre[:,:,0,0])

sx_train = train_X
sy_train = train_Y
print("train_size:", sx_train.shape)
sx_test = val_X
sy_test = val_Y
print("test_size:", sx_test.shape)
sx_pre = x_pre
sy_pre = y_pre
print("pre_size:", sx_pre.shape)

HtoL = True
device_id = 2
x_ratio = 0.01
batch_size = 128

count = int(len(sx_pre)/2)
interval = int(count * x_ratio)
x_arrange = np.arange(0, count, interval)


def eval_acq_mnist_single(phase, knn_pre_scores, sx_train, sy_train, sx_test, sy_test, sx_pre, sy_pre, batch_size, x_ratio, count, epochs=15, HtoL=HtoL, device_id=device_id):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    accs = []
    interval = int(count * x_ratio)
    idxs = np.argsort(knn_pre_scores)
    times = 5
    print(phase+" start:")
    if(HtoL == True):
        print("adding data from Highest to Lowest!")
        idxs = np.flip(idxs, 0)
    else:
        print("adding data from Lowest to Highest!")
    keep_idxs = idxs.tolist()

    for j in range(0, count, interval):
        x_train_keep = np.concatenate((sx_train, sx_pre[keep_idxs[:j]]), axis=0)
        y_train_keep = np.concatenate((sy_train, sy_pre[keep_idxs[:j]]), axis=0)
        clf_knn =  RandomForestClassifier(n_estimators=50, random_state=666)
        clf_knn.fit(x_train_keep, y_train_keep)
        acc = clf_knn.score(sx_test, sy_test) * 100
        accs.append(acc)
        print(x_train_keep.shape, acc)
    print(phase, " :", accs)
    return accs




# evaluate the result
embed_acc = eval_acq_mnist_single("embed", embed_knn_pre_scores, sx_train, sy_train, sx_test, sy_test, sx_pre, sy_pre, batch_size, x_ratio, count, epochs=15, HtoL=HtoL, device_id=device_id)

print("embed_acc: ", embed_acc)
print("x Arrange: ", x_arrange)
