# Src: https://github.com/vahidk/tfrecord

# import torch
# from tfrecord.torch.dataset import TFRecordDataset

# tfrecord_path = "./youtube/train0111.tfrecord"
# index_path = None
# # description = {"image": "byte", "label": "float"}
# description = {
#                 "id": "byte",
#                 "labels": "int",
#                 "rgb": "bytes",
#                 "audio": "bytes",
#                 }
# dataset = TFRecordDataset(tfrecord_path, index_path, None)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# data = next(iter(loader))
# print(data)

# import tensorflow as tf

# filenames = ["./youtube/train0111.tfrecord"]
# raw_dataset = tf.data.TFRecordDataset(filenames)

# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)

# Read and print data:
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['./youtube/train0111.tfrecord'])

_, serialized_example = reader.read(filename_queue)

# Define features
read_features = {
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Movie': tf.VarLenFeature(dtype=tf.string),
    'Movie Ratings': tf.VarLenFeature(dtype=tf.float32),
    'Suggestion': tf.FixedLenFeature([], dtype=tf.string),
    'Suggestion Purchased': tf.FixedLenFeature([], dtype=tf.float32),
    'Purchase Price': tf.FixedLenFeature([], dtype=tf.float32)}

# Extract features from serialized data
read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))