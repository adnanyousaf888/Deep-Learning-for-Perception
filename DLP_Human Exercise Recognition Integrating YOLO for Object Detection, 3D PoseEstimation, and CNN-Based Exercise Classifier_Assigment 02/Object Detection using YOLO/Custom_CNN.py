import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os

#Step 1: 
# loading Data
landmarks_data = pd.read_csv('C:\\Users\outco\\Downloads\\A2_DATA_Q1\\landmarks.csv')
labels_data = pd.read_csv('C:\\Users\outco\\Downloads\\A2_DATA_Q1\\labels.csv')

#Step 2: 
# merge Data
merged_data = pd.merge(landmarks_data, labels_data, on='pose_id')
X = merged_data.drop(['pose_id', 'pose'], axis=1).values  # Landmark coordinates
y = merged_data['pose'].values  # Exercise labels

#step 3:
# normalize Landmarks
X_normalized = (X - np.mean(X,axis=0)) / np.std(X,axis=0)

#Step 4: 
# encode Labels
label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)

#step 5: 
# train-Test Split
X_train,X_test,y_train,y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

#step 6: 
# define CNN Architecture
model = Sequential([
    Dense(128, input_shape=(99,), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes (exercises)
])

#Step 7:
# compiling the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#step 8:
# training the Model
history = model.fit(X_train,y_train,validation_split= 0.1, epochs=100,batch_size=32)

#step 9: evaluate the model
y_pred=np.argmax(model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(y_test , y_pred , target_names=label_encoder.classes_))

#step 10: plotting training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Func to visualize predictions
def visualize_predictions(image_path, landmarks, predicted_label, output_path=None):
    
    ###Overlay landmarks and predicted exercise label on the image.###
    
    #loading image
    image=cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    #get image dimensions
    height, width, _ =image.shape
    #ensure landmark coordinates are properly scaled to image size
    for i in range(0,len(landmarks), 3):
        #scale normalized coordinates from (-1 to 1) to pixel coordinates
        x=int((landmarks[i] * width)/2 + width/2)
        y=int((landmarks[i+1] * height)/2 + height/2)

        #ensuring points are within image bounds
        x=max(0, min(width-1, x))
        y=max(0, min(height-1, y))

        #drawing landmarks
        cv2.circle(image,(x,y),5,(0,255,0), -1)  # Green color

    #overlaying predicted label
    cv2.putText(image, f"Exercise: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # displaying image
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved visualized prediction at {output_path}")
    else:
        cv2.imshow('Prediction', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#visualize Prediction
sample_index=0
sample_landmarks=X_test[sample_index]
predicted_label=label_encoder.inverse_transform([y_pred[sample_index]])[0]

#input image and output path
image_path ='C:\\Users\\outco\\Downloads\\pushup_down.jpg'
output_path='C:\\Users\\outco\\Downloads\\visualized_prediction.jpg'

#calling function
visualize_predictions(image_path, sample_landmarks, predicted_label, output_path)

