import os
import numpy as np
import cv2
import pywt
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Layer, Conv2D, Dropout, MaxPool2D, UpSampling2D, concatenate, Add, Multiply, BatchNormalization
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Define EncoderBlock
class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = None
        self.drop = None
        self.c2 = None
        self.pool = None if not pooling else MaxPool2D()

    def build(self, input_shape):
        self.c1 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.drop = Dropout(self.rate)
        self.c2 = Conv2D(self.filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')
        super(EncoderBlock, self).build(input_shape)

    def call(self, X):
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 'filters': self.filters,
            'rate': self.rate, 'pooling': self.pooling
        }

# Define DecoderBlock
class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'filters': self.filters, 'rate': self.rate}

# Define AttentionGate
class AttentionGate(Layer):
    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()

    def call(self, X):
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'filters': self.filters, 'bn': self.bn}
custom_objects = {
        'EncoderBlock': EncoderBlock,
        'DecoderBlock': DecoderBlock,
        'AttentionGate': AttentionGate
    }
# Load the model with custom objects
with CustomObjectScope(custom_objects):
    new_segmentation_model = load_model("models/segmentation5.h5")
        # Recompile to build metrics and suppress warning
    new_segmentation_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=[MeanIoU(num_classes=2), 'accuracy']
        )



# Load segmentation and classification models
segmentation_model = tf.keras.models.load_model("models/segmentation1_h5.h5", compile=False)
segmentation_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classification_model = tf.keras.models.load_model("models/classification1_h5.h5")

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to apply CLAHE
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# Function to apply Variant-Enhanced Mechanism
def apply_variant_enhancement(image, alpha=2.0, beta=-1.0):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return np.clip(cv2.addWeighted(image, alpha, blurred, beta, 0), 0, 255).astype(np.uint8)

# Function to apply Wavelet Features & Contrast Enhancement
def apply_wavelet_contrast_enhancement(image, enhancement_factor=1.5):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    LH *= enhancement_factor
    HL *= enhancement_factor
    HH *= enhancement_factor
    return np.clip(pywt.idwt2((LL, (LH, HL, HH)), 'haar'), 0, 255).astype(np.uint8)

# Function to apply Fuzzy Enhancement
def apply_fuzzy_enhancement(image):
    image = image.astype(np.float64)
    avmin, avmax = np.min(image), np.max(image)
    am = (image - avmin) / (avmax - avmin)
    amm = np.where(am <= 0.5, 2 * (am ** 2), 1 - 2 * ((1 - am) ** 2))
    return np.stack([(avmin + (amm * (avmax - avmin)))] * 3, axis=-1).astype(np.uint8)

# Image Preprocessing Pipeline
def preprocess_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (size, size))
    image = apply_clahe(image)
    image = apply_variant_enhancement(image)
    image = apply_wavelet_contrast_enhancement(image)
    return apply_fuzzy_enhancement(image)

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess images
    segmentation_input = preprocess_image(file_path, 256)
    classification_input = preprocess_image(file_path, 224)

    # Model Predictions
    segmentation_input = np.expand_dims(segmentation_input, axis=[0, -1]) / 255.0
    segmentation_output = new_segmentation_model.predict(segmentation_input)[0]

    classification_input = np.expand_dims(classification_input, axis=0) / 255.0
    classification_result = classification_model.predict(classification_input)
    predicted_class = np.argmax(classification_result)

    class_labels = {0: "Normal", 1: "Benign", 2: "Malignant"}
    predicted_class_name = class_labels.get(predicted_class, "Unknown")

    # Overlay segmentation mask on original image
    mask = (segmentation_output * 255).astype(np.uint8)
    original = cv2.imread(file_path)

    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    # Ensure both images have the same size
    mask_colored = cv2.resize(mask_colored, (original.shape[1], original.shape[0]))

    # Ensure both images have the same number of channels
    if len(original.shape) == 2:  # If original is grayscale
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    if len(mask_colored.shape) == 2:  # If mask is grayscale
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_GRAY2BGR)

    # Now perform blending
    overlayed_image = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)

    # Save output
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay_' + filename)
    cv2.imwrite(overlay_path, overlayed_image)
    print(predicted_class_name)
    print(str(predicted_class_name))

    return jsonify({'original': filename, 'overlay': 'overlay_' + filename, 'class_result': str(predicted_class_name)})# Serve static files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
