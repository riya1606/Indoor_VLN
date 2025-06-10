# Indoor Navigation Vision-Language Model (VLM)

The Indoor Navigation Vision-Language Model (VLM) is a sophisticated multimodal deep learning framework designed for **visual question answering** and **scene graph reasoning** in indoor environments. Built upon state-of-the-art vision and language models, it integrates synthetic scenes from **AI2-THOR**, real-world datasets such as **COCO** and **TextVQA**, and advanced object detection pipelines to deliver robust and accurate scene understanding.

---

## Key Features
- **Multimodal Reasoning**: Seamlessly integrates image and text inputs to enable comprehensive visual and contextual reasoning.
- **Scene Graph Generation**: Constructs detailed graphs capturing objects and their spatial relationships (e.g., `left_of`, `right_of`, `near`).
- **Robust Object Detection**:
  - **Primary**: Utilizes AI2-THOR metadata for precise object detection in synthetic scenes.
  - **Secondary**: Employs SAM2 segmentation for real-world images.
  - **Fallback**: Implements color-based segmentation to ensure reliability when SAM2 detection fails.
- **Dataset Versatility**: Supports AI2-THOR, COCO, and TextVQA datasets for diverse training scenarios.
- **Memory Optimization**: Employs batch-wise preprocessing with disk persistence to handle large datasets efficiently.
- **Progress Tracking**: Provides real-time feedback during preprocessing and training using `tqdm` progress bars.

---

## Datasets
The model leverages the following datasets to ensure robust performance across synthetic and real-world environments:
- **AI2-THOR**: Synthetic indoor scenes with comprehensive ground-truth metadata, including object types and positions, ideal for scene graph generation and navigation tasks.
- **COCO**: Real-world images with detailed object annotations, filtered for indoor categories (e.g., chair, table, bed, couch).
- **TextVQA**: Real-world images paired with text-based questions and answers to enhance visual question answering capabilities.

---

## Model Architecture
### Vision Backbone
- **Vision Transformer (ViT)**: Extracts high-level image features for robust visual processing.
- **ZoeDepth**: Provides single-view depth estimation to enhance spatial understanding.

### Object Detection Pipeline
- **Primary Detection**: Leverages AI2-THOR metadata for synthetic images.
- **Secondary Detection**: Utilizes SAM2 for segmentation in real-world images.
- **Fallback Mechanism**: Employs color-based segmentation when metadata or SAM2 detection is unavailable or fails.

### Language Model
- **T5 (Text-to-Text Transfer Transformer)**: Encodes input questions and generates precise answers.

### Scene Graph Representation
- **Nodes**: Represent detected objects in the scene.
- **Edges**: Capture spatial relationships between objects (e.g., `left_of`, `right_of`, `near`).

### Cross-Modal Integration
A cross-attention mechanism fuses vision and language features to generate contextually accurate answers.

---

## Training Workflow
### Preprocessing
- Images are processed in batches (default size: 100) to optimize memory usage.
- Processed batches are cached to disk as `processed_batch_*.pt` files for efficient reuse.
- Progress is monitored with `tqdm` progress bars for transparency.

### Training
- Built on **PyTorch** and **HuggingFace's Trainer** for streamlined and scalable training.
- Displays real-time progress for batches and epochs using `tqdm`.
- Supports resuming from preprocessed batches to avoid redundant computation.

### Fallback Mechanisms
- In cases of object detection failure (e.g., SAM2 returns no objects), the model automatically falls back to metadata-based or color-based detection to maintain robustness.

---

## Usage Instructions
### Preprocessing and Training
To preprocess data and train the model, execute the main script:
```bash
python indoor_navigation_vlm.py
```
- The script supports resuming training by loading cached `processed_batch_*.pt` files, ensuring no preprocessing is repeated unnecessarily.

### Inference
Perform inference using the provided Python API or script:
```python
from indoor_navigation_vlm import NavigationVLM

vlm = NavigationVLM("config.yaml")
test_image = "data/test_image.png"
question = "What objects are present in this image?"
answer = vlm.query(test_image, question)
print("Answer:", answer)
```
Alternatively, run the inference script:
```bash
python test_vlm_inference.py
```

---

## File Structure
- `indoor_navigation_vlm.py`: Core script containing the model, dataset handling, and training pipeline.
- `test_vlm_inference.py`: Dedicated script for inference and testing.
- `processed_batch_*.pt`: Automatically generated files storing preprocessed data batches.
- `config.yaml`: Configuration file specifying dataset paths and hyperparameters.

---

## Technical Notes
- **Memory Efficiency**: Batch-wise preprocessing and disk caching enable training on large datasets with constrained RAM or VRAM.
- **Extensibility**: Modular design allows for easy integration of new datasets or model components.
- **Robustness**: Comprehensive fallback mechanisms ensure reliable performance even with incomplete metadata or detection failures.
- **Resumability**: Training can resume seamlessly from cached preprocessed data.

---

## Citation
If you use this codebase, please cite the following foundational works:
- Vision Transformer (ViT)
- Text-to-Text Transfer Transformer (T5)
- ZoeDepth
- Segment Anything Model 2 (SAM2)
- AI2-THOR

---

## Acknowledgements
This project builds upon the following resources:
- **AI2-THOR**: For synthetic indoor scene data.
- **COCO**: For real-world image annotations.
- **TextVQA**: For text-based visual question answering data.
- **ViT**: For vision feature extraction.
- **T5**: For language modeling.
- **ZoeDepth**: For depth estimation.
- **SAM2**: For advanced segmentation.

---

## Example Queries
- "What objects are present in this image?"
- "Is the table positioned to the right of the chair?"
- "What is the distance between the sofa and the window?"

---

