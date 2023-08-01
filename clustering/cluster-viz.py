import os
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

def get_classwise_image_path(root_dir):
    image_paths = dict()
    classes = os.listdir(root_dir)
    for cls in classes:
        image_paths[cls] = []
        class_dir = os.path.join(root_dir, cls)
        images = os.listdir(class_dir)
        for image_name in images:
            img_path = os.path.join(class_dir, image_name)
            image_paths[cls].append(img_path)
    return image_paths

IMG_ROOT_DIR = 'clusters'
image_paths_dict = get_classwise_image_path(IMG_ROOT_DIR)

	
IMG_WIDTH, IMG_HEIGHT = (128, 128)

def load_and_resize_image(img_path, width, height):
    img = Image.open(img_path).resize((width, height))
    return img

def show_class_sample(image_path_dic, fig_size=(15, 6)):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=fig_size
        )
    list_axes = list(axes.flat)
    classes = list(image_path_dic.keys())
    for i, ax in enumerate(list_axes): 
        img = load_and_resize_image(image_path_dic[classes[i]][0], 
            IMG_WIDTH, 
            IMG_HEIGHT)
        ax.imshow(img)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(classes[i])
    fig.suptitle("Iznik Dataset Samples", fontsize=15)
    plt.show()
    return

show_class_sample(image_paths_dict)

def load_model_and_preprocess_func(input_shape, model_family, model_name):  
     
    # Models will be loaded wth pre-trainied `imagenet` weights.
    model = getattr(tf.keras.applications, model_name)(input_shape=input_shape, 
        weights="imagenet", 
        include_top=False)
     
    preprocess  = getattr(tf.keras.applications, model_family).preprocess_input
    return model, preprocess

def get_feature_extractor(model):
    inputs = model.inputs
    x = model(inputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    feat_ext = tf.keras.Model(inputs=inputs, outputs=outputs, 
        name="feature_extractor")
    return feat_ext

IMAGE_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
MODEL_FAMILY = "resnet"
MODEL_NAME   = "ResNet101"
model, preprocess= load_model_and_preprocess_func(IMAGE_SHAPE, 
    MODEL_FAMILY, 
    MODEL_NAME)

feat_ext_model = get_feature_extractor(model)
print(feat_ext_model.summary())

def extract_features(input, model, preprocess):
     
    # Pre-process the input image.
    x = preprocess(input)
 
    # Generate predictions.
    preds = model.predict(x)
 
    return preds[0]

def load_image_for_inference(image_path, img_shape):
     
    # Load the image.
    image = tf.io.read_file(image_path)
     
    # Convert the image from bytes to an image tensor.
    x = tf.image.decode_image(image, channels=img_shape[2])
     
    # Resize image to the input shape required by the model.
    x = tf.image.resize(x, (img_shape[0], img_shape[1]))
     
    # Add a dimension for an image batch representation.
    x = tf.expand_dims(x, axis=0)
 
    return x

def get_images_labels_features(image_paths_dict, feature_extractor, preprocess):
    images = []
    labels = []
    features = []
 
    for cls in image_paths_dict:
        image_paths = image_paths_dict[cls]
        for img_path in image_paths:
            labels.append(cls)
            img = load_and_resize_image(img_path, IMG_WIDTH, IMG_HEIGHT)
            images.append(img)
            img_for_infer = load_image_for_inference(img_path, IMAGE_SHAPE)
            feature = extract_features(img_for_infer, 
                feature_extractor, 
                preprocess)
            features.append(feature)
    return images, labels, features

images, labels, features = get_images_labels_features(image_paths_dict, feat_ext_model, preprocess)

def create_sprite_image(pil_images, save_path):
    # Assuming all images have the same width and height
    img_width, img_height = pil_images[0].size
 
    # create a master square images
    row_coln_count = int(np.ceil(np.sqrt(len(pil_images))))
    master_img_width = img_width * row_coln_count
    master_img_height = img_height * row_coln_count
 
    master_image = Image.new(
        mode = 'RGBA',
        size = (master_img_width, master_img_height),
        color = (0, 0, 0, 0)
    )
 
    for i, img in enumerate(pil_images):
        div, mod = divmod(i, row_coln_count)
        w_loc = img_width * mod
        h_loc = img_height * div
        master_image.paste(img, (w_loc, h_loc))
 
    master_image.convert('RGB').save(save_path, transparency=0)
    return

def write_embedding(log_dir, pil_images, features, labels):
    """Writes embedding data and projector configuration to the logdir."""
    metadata_filename = "metadata.tsv"
    tensor_filename = "features.tsv"
    sprite_image_filename = "sprite.jpg"
 
 
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, metadata_filename), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))
    with open(os.path.join(log_dir, tensor_filename), "w") as f:
        for tensor in features:
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))
 
    sprite_image_path = os.path.join(log_dir, sprite_image_filename)
 
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # Label info.
    embedding.metadata_path = metadata_filename
    # Features info.
    embedding.tensor_path = tensor_filename
    # Image info.
    create_sprite_image(pil_images, sprite_image_path)
    embedding.sprite.image_path = sprite_image_filename
    # Specify the width and height of a single thumbnail.
    img_width, img_height = pil_images[0].size
    embedding.sprite.single_image_dim.extend([img_width, img_height])
    # Create the configuration file.
    projector.visualize_embeddings(log_dir, config)
     
    return

LOG_DIR = os.path.join('logs', MODEL_NAME)
write_embedding(LOG_DIR, images, features, labels)