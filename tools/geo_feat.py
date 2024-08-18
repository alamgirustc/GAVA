import numpy as np
import os
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to extract geometric features
def extract_geometric_features(bbox_file):
    try:
        # Load bounding box data
        bboxes = np.load(bbox_file)

        # Initialize an array to hold the geometric features
        num_rois = bboxes.shape[0]
        if not (10 <= num_rois <= 100):
            raise ValueError(f"Expected between 10 and 100 ROIs, but found {num_rois} ROIs in {bbox_file}")

        geo_features = np.zeros((num_rois, 6))  # 6 features: x, y, w, h, aspect_ratio, area

        for i in range(num_rois):
            x, y, w, h = bboxes[i]
            aspect_ratio = w / h if h != 0 else 0  # Avoid division by zero
            area = w * h
            geo_features[i] = [x, y, w, h, aspect_ratio, area]

        return geo_features
    except Exception as e:
        print(f"Error processing {bbox_file}: {e}")
        return None

# Function to normalize geometric features
def normalize_geometric_features(geo_features):
    if geo_features is None:
        return None
    # Normalize each feature (column) to [0, 1] range
    min_vals = geo_features.min(axis=0)
    max_vals = geo_features.max(axis=0)
    normalized_features = (geo_features - min_vals) / (max_vals - min_vals)

    return normalized_features

# Function to preprocess .npz files and integrate geometric features
def preprocess_npz_files(npz_folder, bbox_folder, geo_output_folder, combined_output_folder):
    files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
    for file_name in tqdm(files, desc="Processing files", unit="file"):
        npz_path = os.path.join(npz_folder, file_name)
        bbox_path = os.path.join(bbox_folder, file_name.replace('.npz', '.npy'))

        # Load existing features
        npz_data = np.load(npz_path)
        features = npz_data['feat']

        num_rois = features.shape[0]
        if not (10 <= num_rois <= 100):
            print(f"Expected between 10 and 100 ROIs in features, but found {num_rois} in {npz_path}. Skipping file.")
            continue

        # Extract geometric features
        geo_features = extract_geometric_features(bbox_path)

        if geo_features is not None:
            # Normalize geometric features
            normalized_geo_features = normalize_geometric_features(geo_features)

            # Save normalized geometric features separately
            geo_save_path = os.path.join(geo_output_folder, file_name.replace('.npz', '_geo.npz'))
            np.savez_compressed(geo_save_path, geo_features=normalized_geo_features)
            #print(f"Processed and saved geometric features: {geo_save_path}")

            # Combine existing features with geometric features
            combined_features = np.concatenate((features, normalized_geo_features), axis=1)

            # Save combined features
            combined_save_path = os.path.join(combined_output_folder, file_name)
            np.savez_compressed(combined_save_path, feat=combined_features)
            #print(f"Processed and saved combined features: {combined_save_path}")
        else:
            print(f"Skipping file due to errors: {npz_path}")

# Example usage
npz_folder = r'D:\geometric\mscoco\feature\up_down_100'
bbox_folder = r'D:\geometric\mscoco\feature\up_down_100_box'
geo_output_folder = r'D:\geometric\mscoco\feature\up_down_100_geo'
combined_output_folder = r'D:\geometric\mscoco\feature\up_down_100_comb_geo'

os.makedirs(geo_output_folder, exist_ok=True)
os.makedirs(combined_output_folder, exist_ok=True)

preprocess_npz_files(npz_folder, bbox_folder, geo_output_folder, combined_output_folder)
