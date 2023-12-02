import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou,accu1,kappa1,ce1,oe1
from train import load_data, create_dir
from sklearn.model_selection import train_test_split
H = 256
W = 256
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (H, W))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x
def load_data(path, split=0.3):
    images = sorted(glob(os.path.join(path, "images/", "*.png")))
    masks = sorted(glob(os.path.join(path, "GT/", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (H, W))
    ori_x = x
    x = x/np.max(x)
    x = x.astype(np.int8)
    return ori_x, x

def save_result(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255.0

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255.0 ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading the model """
    with CustomObjectScope({"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss, "accu1": accu1,"kappa1":kappa1,"ce1":ce1,"oe1":oe1}):
        model = tf.keras.models.load_model("/home/srinivas/pattern-recognition/RESUNET/files/model.h5")

    """ Loading the dataset. """
    dataset_path = "/home/srinivas/pattern-recognition/Res_Unet_Test/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    """ Predict the mask and calculate the metrics values """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split("/")[-1]

        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)
        
        """ Predict the mask """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)
        print(y_pred)
        """ Save the image """
        save_image_path = f"/home/srinivas/pattern-recognition/RESUNET/results/{name}"
        
        save_result(ori_x, ori_y, y_pred, save_image_path)
        
        """ Flattening the numpy arrays. """
        y = y.flatten()
        y_pred = y_pred.flatten()
        
        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import cv2

        # Assuming you have two binary images: predicted_img and reference_img
        # Make sure that the images have the same dimensions
        def area_based_metrics(predicted_img, reference_img):
            # Assuming binary images, convert them to boolean arrays
            predicted_img = np.array(predicted_img, dtype=bool)
            reference_img = np.array(reference_img, dtype=bool)

            # Create a confusion matrix
            confusion_mat = confusion_matrix(reference_img.flatten(), predicted_img.flatten())

            # Check if the confusion matrix has the expected shape
            if confusion_mat.shape != (2, 2):
                # Set metrics to default values
                accuracy, kappa, ce, oe, precision, recall, f1_score1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                # Extract true positive, true negative, false positive, and false negative
                tp = confusion_mat[1, 1]
                tn = confusion_mat[0, 0]
                fp = confusion_mat[0, 1]
                fn = confusion_mat[1, 0]

                # Calculate metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                kappa = (tp * tn - fp * fn) / ((tp + fp) * (fn + tn) + (tp + fn) * (fp + tn))
                ce = fp / (tp + fp)  # Commission Error
                oe = fn / (tp + fn)  # Omission Error

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score1 = 2 * precision * recall / (precision + recall)

            return accuracy, kappa, ce, oe, precision, recall, f1_score1
        # Example usage:
        # Load your binary images as NumPy arrays
        predicted_img = y_pred
        reference_img = y
        # Evaluate edge-based metrics
        accuracy, kappa, ce, oe, precision, recall, f1_score1=area_based_metrics(predicted_img, reference_img)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value,accuracy, kappa, ce, oe, precision, recall, f1_score1])
    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    print("modified")



    """ Saving all the results """
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision","accuracy","kappa","ce","oe","precision","recall", "f1_score"])
    df.to_csv("/home/srinivas/pattern-recognition/RESUNET/files/score.csv")