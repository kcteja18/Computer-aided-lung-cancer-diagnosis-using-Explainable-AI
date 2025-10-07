import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import vtk

# Combine base model and classification layers
# model = Model(inputs=base_model.input, outputs=predictions)

def get_gradcam(model, img_tensor, target_class_idx, last_conv_layer_name='block5_conv3'):
    # Define gradient model
    grad_model = keras.models.Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, target_class_idx]
    grads = tape.gradient(loss, conv_outputs)
    # print(conv_outputs.shape)
    # Compute guided gradients
    guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    # Get convolution outputs and guided gradients
    conv_outputs = conv_outputs[0]
    # print(conv_outputs.shape)
    guided_grads = guided_grads[0]
    
    # Compute weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    # Compute CAM
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    cam = np.maximum(cam, 0)
    cam /= np.max(cam)
    
    return cam

def visualize_gradcam(model, img_path, target_class_idx, alpha = 0.4,last_conv_layer_name='block5_conv3',):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(122,122))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Generate Grad-CAM heatmap
    heatmap = get_gradcam(model, img_array, target_class_idx, last_conv_layer_name)
    # cov_out = np.array(cov_out)
    # Resize heatmap to the size of the input image
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    # cov_out = cv2.resize(cov_out, (img_array.shape[2], img_array.shape[1]))
    # cov_out = cov_out.reshape((122,122,3))
    # cov_out = np.uint8(255 * cov_out)
    
    # Apply colormap
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    
    # Convert colormap to the same data type as img_array[0]
    colormap = colormap.astype(np.float32) / 255.0
    c1 = colormap * 0.7
    # Superimpose heatmap on the original image
    superimposed_img = cv2.addWeighted(img_array[0], 0.5, colormap, 0.5, 0)

    inp_img_shape = cv2.imread(img_path).shape

    c1 = cv2.resize(c1, (inp_img_shape[0], inp_img_shape[1]))
    superimposed_img = cv2.resize(superimposed_img, (inp_img_shape[0], inp_img_shape[1]))
    
    plt.imsave("./uploads/heatmap.png",c1)
    plt.imsave("./uploads/output.png",superimposed_img)
    
def image_predict(model, img_path, img_size=(122, 122)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    pred = model.predict(img_array)[0]
    pred = [round(float(i),2) for i in pred]
    return sorted(pred, reverse=True), sorted(list(range(3)),key=pred.__getitem__, reverse=True)


def load_dicom_series(folder_path):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(folder_path)
    reader.Update()
    return reader.GetOutput()

# Function to visualize lung cancer in 3D and save the model
def visualize_lung_cancer(dicom_folder_path, output_file):
    # Load DICOM series
    image_data = load_dicom_series(dicom_folder_path)

    # Create contour filter to extract lung tumor
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image_data)
    contour.SetValue(0, 400)  # Adjust threshold value as needed

    # Smooth the contours
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(10)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()

    # Save the rendered model
    writer = vtk.vtkPLYWriter()  # You can also use vtkSTLWriter if you prefer STL format
    writer.SetFileName(output_file)
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.Write()