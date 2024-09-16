import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def resize_clip(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))
    x = tf.image.resize(x, (32, 32))
    x = x/255.0
    x = tf.clip_by_value(x, 0., 1.)
    return x

def make_dataset(x, y, f, seed=42):
    x = tf.data.Dataset.from_tensor_slices(x).map(f)
    y = tf.data.Dataset.from_tensor_slices(y)
    return tf.data.Dataset.zip((x, y)).shuffle(1000, seed=seed)

def load_dataset(name, frac=1.0, num_class_to_remove=0):
    if name=="cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        if frac < 1.0:
            x_server, _, y_server, _ = train_test_split(x_server, y_server, train_size=frac, random_state=42)

        if num_class_to_remove > 0:
            x_server = x_server[(y_server>=num_class_to_remove).flatten()]
            y_server = y_server[(y_server>=num_class_to_remove).flatten()]

        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        server_ds = make_dataset(x_server, y_server, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        return client_ds, server_ds
    elif name=="cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        if frac < 1.0:
            x_server, _, y_server, _ = train_test_split(x_server, y_server, train_size=frac, random_state=42)

        if num_class_to_remove > 0:
            x_server = x_server[(y_server>=num_class_to_remove).flatten()]
            y_server = y_server[(y_server>=num_class_to_remove).flatten()]

        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        server_ds = make_dataset(x_server, y_server, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        return client_ds, server_ds
    elif name=="tinyimagenet":
        (x_train, y_train), (x_test, y_test) = load_tiny_imagenet()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        if frac < 1.0:
            x_server, _, y_server, _ = train_test_split(x_server, y_server, train_size=frac, random_state=42)

        if num_class_to_remove > 0:
            x_server = x_server[(y_server>=num_class_to_remove).flatten()]
            y_server = y_server[(y_server>=num_class_to_remove).flatten()]

        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        server_ds = make_dataset(x_server, y_server, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        return client_ds, server_ds
    elif name=="stl10":
        (x_train, y_train), (x_test, y_test) = load_stl10()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        if frac < 1.0:
            x_server, _, y_server, _ = train_test_split(x_server, y_server, train_size=frac, random_state=42)

        if num_class_to_remove > 0:
            x_server = x_server[(y_server>=num_class_to_remove).flatten()]
            y_server = y_server[(y_server>=num_class_to_remove).flatten()]

        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        server_ds = make_dataset(x_server, y_server, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        return client_ds, server_ds


def load_single_batch(name, batch_size=128):
    if name=="cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, _, y_client, _ = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.), seed=420)
        return client_ds.batch(batch_size).take(1)
    elif name=="cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, _, y_client, _ = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.), seed=420)
        return client_ds.batch(batch_size).take(1)
    elif name=="tinyimagenet":
        (x_train, y_train), (x_test, y_test) = load_tiny_imagenet()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, _, y_client, _ = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.), seed=420)
        return client_ds.batch(batch_size).take(1)
    elif name=="stl10":
        (x_train, y_train), (x_test, y_test) = load_stl10()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, _, y_client, _ = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.), seed=420)
        return client_ds.batch(batch_size).take(1)

def sample_batch_with_aligned_labels(x_ds_by_labels, labels):
    res_x = np.stack([x_ds_by_labels[y][np.random.choice(len(x_ds_by_labels[y]))] for y in labels])
    return res_x, labels

def load_dataset_with_label_alignment(name):
    if name=="cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        x_server = np.clip(x_server/255.0, 0.0, 1.0)
        # server_ds = make_dataset(x_server, y_server, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        num_class = 10
        server_x_ds_by_labels = [None] * num_class
        # select labels
        for i in range(num_class):
            server_x_ds_by_labels[i] = x_server[(y_server==i).flatten()]
        return client_ds, server_x_ds_by_labels
    elif name=="cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        x_server = np.clip(x_server/255.0, 0.0, 1.0)
        num_class = 100
        server_x_ds_by_labels = [None] * num_class
        # select labels
        for i in range(num_class):
            server_x_ds_by_labels[i] = x_server[(y_server==i).flatten()]
        return client_ds, server_x_ds_by_labels
    elif name=="tinyimagenet":
        (x_train, y_train), (x_test, y_test) = load_tiny_imagenet()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        x_server = np.clip(x_server/255.0, 0.0, 1.0)
        num_class = 200
        server_x_ds_by_labels = [None] * num_class
        # select labels
        for i in range(num_class):
            server_x_ds_by_labels[i] = x_server[(y_server==i).flatten()]
        return client_ds, server_x_ds_by_labels
    elif name=="stl10":
        (x_train, y_train), (x_test, y_test) = load_stl10()
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        y = np.concatenate((y_train, y_test))
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)
        client_ds = make_dataset(x_client, y_client, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        x_server = np.clip(x_server/255.0, 0.0, 1.0)
        num_class = 10
        server_x_ds_by_labels = [None] * num_class
        # select labels
        for i in range(num_class):
            server_x_ds_by_labels[i] = x_server[(y_server==i).flatten()]
        return client_ds, server_x_ds_by_labels
    
import imageio

def load_tiny_imagenet():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../data/tiny-imagenet-200/')
    def get_id_dictionary():
        id_dict = {}
        for i, line in enumerate(open( path + 'wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict
    
    def get_class_to_id_dict():
        id_dict = get_id_dictionary()
        all_classes = {}
        result = {}
        for i, line in enumerate(open( path + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])
        return result

    def get_data(id_dict):
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        for key, value in id_dict.items():
            train_data += [imageio.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), mode='RGB') for i in range(500)]
            train_labels_ = np.array([[0]*200]*500)
            train_labels_[:, value] = 1
            train_labels += train_labels_.tolist()

        for line in open( path + 'val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(imageio.imread( path + 'val/images/{}'.format(img_name) ,mode='RGB'))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()
        return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
    
    train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())

    train_labels = np.argmax(train_labels, axis=1).reshape(-1, 1)
    test_labels = np.argmax(test_labels, axis=1).reshape(-1, 1)

    return (train_data, train_labels), (test_data, test_labels)

def load_stl10():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../data/stl10_binary/')


    with open(os.path.join(path, 'train_X.bin'), 'rb') as f:
        train_data = np.fromfile(f, dtype=np.uint8)
        train_data = np.reshape(train_data, (-1, 3, 96, 96))
        train_data = np.transpose(train_data, (0, 3, 2, 1))

    with open(os.path.join(path, 'test_X.bin'), 'rb') as f:
        test_data = np.fromfile(f, dtype=np.uint8)
        test_data = np.reshape(test_data, (-1, 3, 96, 96))
        test_data = np.transpose(test_data, (0, 3, 2, 1))

    with open(os.path.join(path, 'train_y.bin'), 'rb') as f:
        train_labels = np.fromfile(f, dtype=np.uint8).reshape(-1, 1).astype(int) - 1

    with open(os.path.join(path, 'test_y.bin'), 'rb') as f:
        test_labels = np.fromfile(f, dtype=np.uint8).reshape(-1, 1).astype(int) - 1
    
    return (train_data, train_labels), (test_data, test_labels)

def load_dataset_with_hetero_clients(name, number_of_clients=2):
    if name=="cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif name=="cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    elif name=="tinyimagenet":
        (x_train, y_train), (x_test, y_test) = load_tiny_imagenet()
    elif name=="stl10":
        (x_train, y_train), (x_test, y_test) = load_stl10()
    else:
        raise NotImplementedError("Not implemented yet")
    
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test))
    x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5, random_state=42)

    client_ds_list = [None]*number_of_clients
    server_ds_list = [None]*number_of_clients
    
    for i in range(number_of_clients):
        x_client_i = x_client[y_client.flatten() % number_of_clients == i]
        y_client_i = y_client[y_client.flatten() % number_of_clients == i]

        x_server_i = x_server[y_server.flatten() % number_of_clients == i]
        y_server_i = y_server[y_server.flatten() % number_of_clients == i]

        client_ds_i = make_dataset(x_client_i, y_client_i, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        client_ds_list[i] = client_ds_i

        server_ds_i = make_dataset(x_server_i, y_server_i, lambda x: tf.clip_by_value(x/255.0, 0., 1.))
        server_ds_list[i] = server_ds_i

    return client_ds_list, server_ds_list
    