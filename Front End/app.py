import os
import torch
import pydicom
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import timm

app = Flask(__name__)

# Define folder for saving uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'dcm', 'nii'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
def load_weights_skip_mismatch(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

model = timm.create_model('resnet18', pretrained=True, num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
weights_path = r'E:\CodeSpace\Major Project\resnet18.pt'
load_weights_skip_mismatch(model, weights_path, device)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert DICOM to PNG
def convert_dcm_to_png_sagittal(dcm_file_path):
    dicom_data = pydicom.dcmread(dcm_file_path)
    pixel_array = dicom_data.pixel_array
    if len(pixel_array.shape) == 3:
        middle_index = pixel_array.shape[0] // 2
        pixel_array = pixel_array[middle_index]
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
    pixel_array = pixel_array.astype(np.uint8)
    image = Image.fromarray(pixel_array)
    return image

# Predict on the image
def predict(image, model, device, target_size=(256, 256)):
    image = image.resize(target_size)
    image_rgb = np.array(image.convert("RGB"))
    image_transposed = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_transposed).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(image_tensor)
        preds = torch.sigmoid(preds)
    return preds.cpu().numpy()

# Plot predictions on the image
def plot_predictions_with_labels(image, predictions):
    image_rgb = np.array(image.convert("RGB"))
    for i in range(0, len(predictions[0]), 2):
        x = int(predictions[0][i] * image_rgb.shape[1])
        y = int(predictions[0][i + 1] * image_rgb.shape[0])
        cv2.circle(image_rgb, (x, y), radius=10, color=(255, 0, 0), thickness=-1)
        label = f"({x}, {y})"
        font_scale = 0.6
        thickness = 2
        cv2.putText(image_rgb, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis('off')

    # Save the plot to a static folder
    output_image_path = os.path.join('static', 'output_image.png')
    plt.savefig(output_image_path, format="png")
    plt.close()  # Close the figure to avoid warnings
    return output_image_path  # Return the path instead of encoded image


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_image_path)
        try:
            image = convert_dcm_to_png_sagittal(input_image_path)
            preds = predict(image, model, device)
            output_image_path = plot_predictions_with_labels(image, preds)
            output_image_path = plot_predictions_with_labels(image, preds)
            output_image_path = output_image_path.replace("\\", "/")

            # Render result.html with the output image path
            return render_template('result.html', output_image_path=output_image_path, prediction="Your model prediction here")
        except Exception as e:
            return f"Error processing image: {str(e)}"
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
