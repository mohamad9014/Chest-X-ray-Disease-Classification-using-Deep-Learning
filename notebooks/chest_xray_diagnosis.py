import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
import util

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

labels = [
    'Cardiomegaly',
    'Emphysema',
    'Effusion',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Atelectasis',
    'Pneumothorax',
    'Pleural_Thickening',
    'Pneumonia',
    'Fibrosis',
    'Edema',
    'Consolidation'
]


def check_for_leakage(df1, df2, patient_col):
    df1_patients = set(df1[patient_col].unique())
    df2_patients = set(df2[patient_col].unique())
    leakage = len(df1_patients.intersection(df2_patients)) > 0
    return leakage


def get_train_generator(
    df,
    image_dir,
    x_col,
    y_cols,
    shuffle=True,
    batch_size=8,
    seed=1,
    target_w=320,
    target_h=320
):
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )

    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h)
    )

    return generator


def get_test_and_valid_generator(
    valid_df,
    test_df,
    train_df,
    image_dir,
    x_col,
    y_cols,
    sample_size=100,
    batch_size=8,
    seed=1,
    target_w=320,
    target_h=320
):
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image",
        y_col=y_cols,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h)
    )

    batch = raw_train_generator.next()
    data_sample = batch[0]

    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    image_generator.fit(data_sample)

    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h)
    )

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h)
    )

    return valid_generator, test_generator


def compute_class_freqs(labels_array):
    positive_frequencies = np.mean(labels_array, axis=0)
    negative_frequencies = 1 - positive_frequencies
    return positive_frequencies, negative_frequencies


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0

        for i in range(len(pos_weights)):
            loss_pos = -pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)
            loss_neg = -neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)
            loss += K.mean(loss_pos + loss_neg)

        return loss

    return weighted_loss


train_df = pd.read_csv("data/nih/train-small.csv")
valid_df = pd.read_csv("data/nih/valid-small.csv")
test_df = pd.read_csv("data/nih/test.csv")

print("Leakage between train and valid:", check_for_leakage(train_df, valid_df, "PatientId"))
print("Leakage between train and test:", check_for_leakage(train_df, test_df, "PatientId"))
print("Leakage between valid and test:", check_for_leakage(valid_df, test_df, "PatientId"))

image_dir = "./data/nih/images-small/"

train_generator = get_train_generator(train_df, image_dir, "Image", labels)
valid_generator, test_generator = get_test_and_valid_generator(
    valid_df,
    test_df,
    train_df,
    image_dir,
    "Image",
    labels
)

x, y = train_generator.__getitem__(0)
plt.imshow(x[0])
plt.show()

plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
neg_data = pd.DataFrame(
    [{"Class": labels[i], "Label": "Negative", "Value": v} for i, v in enumerate(freq_neg)]
)
data = pd.concat([data, neg_data], ignore_index=True)

plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label", data=data)
plt.show()

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
neg_data = pd.DataFrame(
    [{"Class": labels[i], "Label": "Negative", "Value": v} for i, v in enumerate(neg_contribution)]
)
data = pd.concat([data, neg_data], ignore_index=True)

plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label", data=data)
plt.show()

base_model = DenseNet121(weights="models/nih/densenet.hdf5", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss=get_weighted_loss(pos_weights, neg_weights))

model.load_weights("models/nih/pretrained_model.h5")

predicted_vals = model.predict(test_generator, steps=len(test_generator))
auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)

df = pd.read_csv("data/nih/train-small.csv")
IMAGE_DIR = "data/nih/images-small/"

labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]

util.compute_gradcam(model, "00008270_015.png", IMAGE_DIR, df, labels, labels_to_show)
util.compute_gradcam(model, "00011355_002.png", IMAGE_DIR, df, labels, labels_to_show)
util.compute_gradcam(model, "00029855_001.png", IMAGE_DIR, df, labels, labels_to_show)
util.compute_gradcam(model, "00005410_000.png", IMAGE_DIR, df, labels, labels_to_show)
