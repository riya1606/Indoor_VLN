import os
import yaml
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTModel, ViTImageProcessor, T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Environment settings for stable CUDA operations and tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class MetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir):
        super().__init__()
        self.tokenizer = tokenizer
        self.metrics_file = os.path.join(output_dir, "training_metrics.csv")
        self.metrics = {'epoch': [], 'step': [], 'loss': []}
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.metrics['epoch'].append(state.epoch)
            self.metrics['step'].append(state.global_step)
            self.metrics['loss'].append(logs['loss'])
            pd.DataFrame(self.metrics).to_csv(self.metrics_file, index=False)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            if len(self.metrics['epoch']) > 0:
                pd.DataFrame(self.metrics).to_csv(self.metrics_file, index=False)

class CustomProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.epoch_progress = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.progress_bar = tqdm(total=self.total_steps, desc="Training", unit="step")
        self.epoch_progress = tqdm(total=args.num_train_epochs, desc="Epochs", unit="epoch")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.progress_bar.set_postfix(loss=f"{logs['loss']:.4f}")
                
    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_progress.update(1)
        epoch_num = int(state.epoch)
        output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch_num + 1}")
        os.makedirs(output_dir, exist_ok=True)
        state.trainer.save_model(output_dir)
        
    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()
        self.epoch_progress.close()


class NavigationDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        os.makedirs("data", exist_ok=True)
        # Load datasets from multiple sources
        self.load_ai2thor()
        self.load_textvqa()
        self.load_coco()

    def load_ai2thor(self):
        import ai2thor.controller
        controller = ai2thor.controller.Controller(scene="FloorPlan1", width=512, height=512)
        for i in tqdm(range(self.config['max_scenes']), desc="AI2-THOR"):
            scene_id = f"FloorPlan{i % 30 + 1}"
            controller.reset(scene=scene_id)
            event = controller.step(action="Done")
            rgb = event.frame
            objects = event.metadata["objects"]
            visible_objects = [obj for obj in objects if obj.get('visible', False)]
            image_path = f"data/ai2thor_{scene_id}_{i}.png"
            metadata_path = f"data/ai2thor_{scene_id}_{i}_metadata.json"
            cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # Add screen position if missing for object localization
            for obj in visible_objects:
                if 'screenPosition' not in obj:
                    obj['screenPosition'] = {'x': int(256 + obj['position']['x'] * 100), 'y': int(256 + obj['position']['z'] * 100)}
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(visible_objects, f, indent=2)
            self.data.extend(self.generate_vqa_pairs(image_path, metadata_path, f"ai2thor_{scene_id}_{i}", "ai2thor"))
        controller.stop()

    def generate_vqa_pairs(self, image_path, metadata_path, scene_id, source):
        vqa_pairs = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            objects = json.load(f)
        if len(objects) < 2:
            return vqa_pairs
        # Limit the number of object pairs to avoid excessive combinations
        max_pairs = min(10, len(objects) * (len(objects) - 1) // 2)
        selected_pairs = random.sample([(i, j) for i in range(len(objects)) for j in range(i+1, len(objects))], max_pairs)
        for i, j in selected_pairs:
            obj1, obj2 = objects[i], objects[j]
            obj1_type = obj1.get('objectType', 'unknown')
            obj2_type = obj2.get('objectType', 'unknown')
            pos1 = obj1.get('position', {})
            pos2 = obj2.get('position', {})
            dx = pos1.get('x', 0) - pos2.get('x', 0)
            dz = pos1.get('z', 0) - pos2.get('z', 0)
            direction = 'left' if dx < -0.5 else 'right' if dx > 0.5 else 'near'
            if abs(dz) > abs(dx):
                direction = 'in front of' if dz < -0.5 else 'behind' if dz > 0.5 else 'near'
            question = f"Is the {obj1_type} {direction} of the {obj2_type}?"
            answer = "Yes" if direction in ['left', 'right', 'in front of', 'behind'] else "No"
            vqa_pairs.append({'image_path': image_path, 'metadata_path': metadata_path, 'question': question, 'answer': answer, 'source': source})
            distance = np.sqrt(dx**2 + dz**2)
            question = f"How far is the {obj1_type} from the {obj2_type} in meters?"
            answer = f"Approximately {distance:.2f} meters"
            vqa_pairs.append({'image_path': image_path, 'metadata_path': metadata_path, 'question': question, 'answer': answer, 'source': source})
        return vqa_pairs

    def load_textvqa(self):
        with open(self.config['textvqa_path'], 'r', encoding='utf-8') as f:
            textvqa_data = json.load(f)
        data_list = textvqa_data.get('data', [])
        random.seed(42)
        random.shuffle(data_list)
        train_data = data_list[:int(0.8 * len(data_list))]
        for item in tqdm(train_data[:self.config['max_textvqa']], desc="TextVQA"):
            image_path = os.path.join(self.config['textvqa_image_dir'], 'train_images', item['image_id'] + '.jpg')
            if os.path.exists(image_path):
                self.data.append({'image_path': image_path, 'question': item['question'], 'answer': item['answers'][0], 'source': 'textvqa'})

    def load_coco(self):
        with open(self.config['coco_annotations'], 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        # Filter for indoor scenes based on object categories
        indoor_categories = ['chair', 'table', 'bed', 'couch', 'toilet', 'tv', 'microwave', 'oven', 'refrigerator']
        indoor_image_ids = set(ann['image_id'] for ann in coco_data['annotations'] 
                               if any(cat['name'] in indoor_categories for cat in coco_data['categories'] if cat['id'] == ann['category_id']))
        indoor_images = [img for img in coco_data['images'] if img['id'] in indoor_image_ids][:self.config['max_coco']]
        for img in tqdm(indoor_images, desc="COCO"):
            objects = [ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]
            if objects:
                img_path = os.path.join(self.config['coco_dir'], img['file_name'])
                if os.path.exists(img_path):
                    rgb = cv2.imread(img_path)
                    depth = np.zeros_like(rgb[:, :, 0])
                    object_data = []
                    for ann in objects:
                        category_id = ann['category_id']
                        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), f"obj_{category_id}")
                        object_data.append({'id': ann['id'], 'name': category_name, 'centroid': [ann['bbox'][0] + ann['bbox'][2]/2, ann['bbox'][1] + ann['bbox'][3]/2]})
                    self.data.extend(self.generate_vqa_pairs_coco(rgb, depth, object_data, img['file_name'], "coco"))

    def generate_vqa_pairs_coco(self, rgb, depth, objects, scene_id, source):
        vqa_pairs = []
        if len(objects) < 2:
            return vqa_pairs
        max_pairs = min(10, len(objects) * (len(objects) - 1) // 2)
        selected_pairs = random.sample([(i, j) for i in range(len(objects)) for j in range(i+1, len(objects))], max_pairs)
        for i, j in selected_pairs:
            obj1, obj2 = objects[i], objects[j]
            dx = obj1['centroid'][0] - obj2['centroid'][0]
            dy = obj1['centroid'][1] - obj2['centroid'][1]
            direction = 'left' if dx < -50 else 'right' if dx > 50 else 'near'
            if abs(dy) > abs(dx):
                direction = 'above' if dy < -50 else 'below' if dy > 50 else 'near'
            image_path = f"data/{source}_{scene_id.replace('/', '_')}_{i}_{j}.png"
            cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            question = f"Is {obj1['name']} {direction} of {obj2['name']}?"
            answer = "Yes" if direction in ['left', 'right', 'above', 'below'] else "No"
            vqa_pairs.append({'image_path': image_path, 'question': question, 'answer': answer, 'source': source})
            distance = np.sqrt(dx**2 + dy**2) / 100
            question = f"How far is {obj1['name']} from {obj2['name']} in meters?"
            answer = f"Approximately {distance:.2f} meters"
            vqa_pairs.append({'image_path': image_path, 'question': question, 'answer': answer, 'source': source})
        return vqa_pairs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class NavigationVLM:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_cache = {}
        # Initialize models for depth estimation, segmentation, and vision-language tasks
        self.zoe_model = self.load_zoe_depth()
        self.sam_model = self.load_sam_model()
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(self.device)
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.dataset = NavigationDataset(self.config)

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_zoe_depth(self):
        config = get_config("zoedepth", "infer")
        model = build_model(config)
        state_dict = torch.hub.load_state_dict_from_url("https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt", progress=True)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def load_sam_model(self):
        # Configure SAM2 for automatic mask generation
        sam = build_sam2(
            config_file="sam2_hiera_l.yaml",
            checkpoint=self.config['sam_checkpoint'],
            device=self.device,
            apply_postprocessing=True
        )
        return SAM2AutomaticMaskGenerator(sam, pred_iou_thresh=0.5, stability_score_thresh=0.5, min_mask_region_area=10, points_per_side=32, box_nms_thresh=0.5)

    def clear_memory(self):
        self._image_cache.clear()
        torch.cuda.empty_cache()

    def detect_objects_color_based(self, img_np, depth_map=None):
        objects = []
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        height, width = img_np.shape[:2]
        # Define color ranges for common indoor objects
        color_ranges = [
            ("table", np.array([10, 50, 50]), np.array([30, 255, 200])),
            ("floor", np.array([0, 0, 100]), np.array([180, 30, 255])),
            ("wall", np.array([0, 0, 50]), np.array([180, 30, 200])),
            ("chair", np.array([90, 50, 50]), np.array([150, 255, 255])),
            ("sofa", np.array([0, 50, 50]), np.array([10, 255, 255])),
            ("cabinet", np.array([20, 50, 50]), np.array([40, 255, 200])),
            ("refrigerator", np.array([0, 0, 150]), np.array([180, 30, 255])),
            ("sink", np.array([90, 10, 100]), np.array([130, 50, 255])),
            ("window", np.array([0, 0, 200]), np.array([180, 30, 255])),
            ("door", np.array([15, 30, 80]), np.array([35, 255, 200]))
        ]
        obj_id = 0
        for name, lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    centroid = [x + w/2, y + h/2]
                    centroid_depth = depth_map[min(int(centroid[1]), depth_map.shape[0]-1), min(int(centroid[0]), depth_map.shape[1]-1)] if depth_map is not None else y / height
                    objects.append({'id': obj_id, 'bbox': [x, y, w, h], 'centroid': centroid, 'depth': float(centroid_depth), 'label': name, 'area': cv2.contourArea(contour)})
                    obj_id += 1
        # Fallback for when no objects are detected
        if not objects:
            regions = [(0, 0, width//2, height//2, "wall"), (width//2, 0, width//2, height//2, "window"), 
                       (0, height//2, width//2, height//2, "floor"), (width//2, height//2, width//2, height//2, "furniture")]
            for i, (x, y, w, h, label) in enumerate(regions):
                centroid = [x + w/2, y + h/2]
                centroid_depth = depth_map[min(int(centroid[1]), depth_map.shape[0]-1), min(int(centroid[0]), depth_map.shape[1]-1)] if depth_map is not None else y / height
                objects.append({'id': i, 'bbox': [x, y, w, h], 'centroid': centroid, 'depth': float(centroid_depth), 'label': label, 'area': w * h})
        return objects

    def process_image(self, image_path):
        if image_path in self._image_cache:
            return self._image_cache[image_path]
        if not os.path.exists(image_path):
            return None
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        # Generate depth map using ZoeDepth
        with torch.no_grad():
            depth = self.zoe_model.infer(img_tensor).squeeze().cpu().numpy()
        metadata_path = image_path.replace('.png', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_objects = json.load(f)
            objects = []
            for i, obj in enumerate(metadata_objects):
                screen_position = obj.get('screenPosition', {'x': 0, 'y': 0})
                x, y = screen_position.get('x', 0), screen_position.get('y', 0)
                if 0 <= x <= img_np.shape[1] and 0 <= y <= img_np.shape[0]:
                    objects.append({'id': i, 'bbox': [x-25, y-25, 50, 50], 'centroid': [x, y], 'depth': obj.get('position', {}).get('z', 0), 'label': obj.get('objectType', f'obj_{i}')})
        else:
            # Use SAM2 for object segmentation if metadata is unavailable
            masks = self.sam_model.generate(img_np)
            if masks:
                objects = [{'id': i, 'bbox': mask['bbox'], 'centroid': [mask['bbox'][0] + mask['bbox'][2]/2, mask['bbox'][1] + mask['bbox'][3]/2], 
                           'depth': float(depth[min(int(mask['bbox'][1] + mask['bbox'][3]/2), depth.shape[0]-1), min(int(mask['bbox'][0] + mask['bbox'][2]/2), depth.shape[1]-1)]), 
                           'label': f'obj_{i}'} for i, mask in enumerate(masks)]
            else:
                objects = self.detect_objects_color_based(img_np, depth)
        inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vit_outputs = self.vit_model(**inputs)
        self._image_cache[image_path] = {'image_path': image_path, 'depth_map': depth, 'objects': objects, 'vit_features': vit_outputs.last_hidden_state.detach()}
        return self._image_cache[image_path]

    def generate_scene_graph(self, image_data):
        objects = image_data['objects']
        nodes = [{'id': obj['id'], 'label': obj['label'], 'position': obj['centroid'], 'depth': obj['depth'], 'size': [obj['bbox'][2], obj['bbox'][3]]} for obj in objects]
        edges = []
        # Create edges based on spatial relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:]):
                dx = obj1['centroid'][0] - obj2['centroid'][0]
                dy = obj1['centroid'][1] - obj2['centroid'][1]
                ddepth = obj1['depth'] - obj2['depth']
                distance_3d = np.sqrt(dx**2 + dy**2 + ddepth**2)
                distance_2d = np.sqrt(dx**2 + dy**2)
                if abs(ddepth) > max(abs(dx), abs(dy)) and abs(ddepth) > 0.5:
                    relation = 'in_front_of' if ddepth < 0 else 'behind'
                elif abs(dx) > abs(dy) and abs(dx) > 20:
                    relation = 'left_of' if dx < 0 else 'right_of'
                elif abs(dy) > 20:
                    relation = 'above' if dy < 0 else 'below'
                else:
                    relation = 'near'
                edges.append({'source': obj1['id'], 'target': obj2['id'], 'relation': relation, 'distance_2d': float(distance_2d), 'distance_3d': float(distance_3d)})
        return {'nodes': nodes, 'edges': edges}

    def visualize_scene_graph(self, image_path, scene_graph):
        img = cv2.imread(image_path)
        for node in scene_graph['nodes']:
            x, y = int(node['position'][0]), int(node['position'][1])
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, node['label'], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for edge in scene_graph['edges']:
            source = next(n for n in scene_graph['nodes'] if n['id'] == edge['source'])
            target = next(n for n in scene_graph['nodes'] if n['id'] == edge['target'])
            sx, sy = int(source['position'][0]), int(source['position'][1])
            tx, ty = int(target['position'][0]), int(target['position'][1])
            cv2.line(img, (sx, sy), (tx, ty), (255, 0, 0), 1)
            cv2.putText(img, edge['relation'], ((sx + tx) // 2, (sy + ty) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(f"debug_{os.path.basename(image_path)}", img)

    def preprocess_dataset(self):
        all_data = []
        batch_size = 100
        for batch_idx in tqdm(range((len(self.dataset) + batch_size - 1) // batch_size), desc="Preprocessing"):
            batch_data = []
            for idx in range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(self.dataset))):
                example = self.dataset[idx]
                image_data = self.process_image(example['image_path'])
                if image_data is None:
                    continue
                vit_features = image_data['vit_features'].squeeze(0).cpu()
                # Tokenize question and answer for T5 model
                inputs = self.t5_tokenizer(f"Question: {example['question']}\nScene info: {example['source']}\nAnswer:", 
                                         return_tensors="pt", padding="max_length", max_length=128, truncation=True)
                labels = self.t5_tokenizer(example['answer'], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
                batch_data.append({
                    'input_ids': inputs.input_ids.squeeze().cpu(),
                    'attention_mask': inputs.attention_mask.squeeze().cpu(),
                    'labels': labels.input_ids.squeeze().cpu(),
                    'decoder_attention_mask': labels.attention_mask.squeeze().cpu(),
                    'vit_features': vit_features,
                    'source': example['source']
                })
            torch.save(batch_data, f"processed_batch_{batch_idx}.pt")
            all_data.extend(batch_data)
            self.clear_memory()
        return all_data

    def load_preprocessed_batches(self):
        import glob
        batch_files = sorted(glob.glob("processed_batch_*.pt"))
        all_data = []
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            all_data.extend(torch.load(batch_file))
        return all_data
    
    # Computes accuracy for model predictions
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        pred_texts = [self.t5_tokenizer.decode(pred, skip_special_tokens=True).lower().strip() for pred in predictions]
        label_texts = [self.t5_tokenizer.decode(label, skip_special_tokens=True).lower().strip() for label in labels]
        exact_matches = [1 if pred == label else 0 for pred, label in zip(pred_texts, label_texts)]
        accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        return {'accuracy': accuracy}

    def train(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        batch_files = [f for f in os.listdir(".") if f.startswith("processed_batch_") and f.endswith(".pt")]
        train_dataset = self.load_preprocessed_batches() if batch_files else self.preprocess_dataset()
        self.clear_memory()
        train_size = int(0.9 * len(train_dataset))
        train_data, eval_data = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        
        class PreprocessedDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Configure training parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=3,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            learning_rate=float(self.config['learning_rate']),
            warmup_steps=100,
            weight_decay=0.01,
            fp16=True,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            gradient_checkpointing=True,
            optim="adamw_torch",
            save_strategy="epoch",
            save_safetensors=False,
            evaluation_strategy="epoch",
            dataloader_drop_last=True,
        )
        
        # Define model with cross-attention between vision and text features
        class CrossAttentionVLM(torch.nn.Module):
            def __init__(self, vit_model, t5_model):
                super().__init__()
                self.vit = vit_model
                self.t5 = t5_model
                self.cross_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=0.1)
                self.fc = torch.nn.Linear(768, self.t5.config.d_model)
            def gradient_checkpointing_enable(self, **kwargs):
                self.t5.gradient_checkpointing_enable()
            def gradient_checkpointing_disable(self):
                self.t5.gradient_checkpointing_disable()
            def forward(self, input_ids, attention_mask, vit_features, labels=None, decoder_attention_mask=None):
                input_ids = input_ids.to(self.t5.device)
                attention_mask = attention_mask.to(self.t5.device)
                vit_features = vit_features.to(self.t5.device)
                if labels is not None:
                    labels = labels.to(self.t5.device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.t5.device)
                text_features = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                attn_output, _ = self.cross_attention(query=text_features, key=vit_features, value=vit_features)
                combined = text_features + self.fc(attn_output)
                if labels is not None:
                    return self.t5(inputs_embeds=combined, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
                return self.t5.generate(encoder_outputs=(combined,), attention_mask=attention_mask, max_length=64)
        
        model = CrossAttentionVLM(self.vit_model, self.t5_model).to(self.device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=PreprocessedDataset(train_data),
            eval_dataset=PreprocessedDataset(eval_data),
            compute_metrics=self.compute_metrics,
            callbacks=[CustomProgressCallback(), MetricsCallback(self.t5_tokenizer, output_dir)]
        )
        trainer.state.trainer = trainer
        trainer.train()
        # Save models and tokenizer
        self.t5_model.save_pretrained(os.path.join(output_dir, "t5_model"), safe_serialization=False)
        self.t5_tokenizer.save_pretrained(os.path.join(output_dir, "t5_tokenizer"))
        torch.save(self.vit_model.state_dict(), os.path.join(output_dir, "vit_model.pt"))
        # Plot training metrics
        metrics_df = pd.read_csv(os.path.join(output_dir, "training_metrics.csv"))
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(metrics_df['step'], metrics_df['loss'], label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(metrics_df['step'], metrics_df['accuracy'], label='Accuracy')
        plt.plot(metrics_df['step'], metrics_df['f1'], label='F1 Score')
        plt.xlabel('Steps')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))

    def query(self, image_path, question):
        image_data = self.process_image(image_path)
        if image_data is None:
            return "Could not process the image."
        scene_graph = self.generate_scene_graph(image_data)
        self.visualize_scene_graph(image_path, scene_graph)
        
        # Customize scene info based on question type
        if "objects do you see" in question.lower():
            scene_info = f"Objects: {', '.join(node['label'] for node in scene_graph['nodes'])}"
        elif "spatial relationship" in question.lower():
            scene_info = f"Relationships: {'; '.join(f'{edge['source']} is {edge['relation']} {edge['target']}' for edge in scene_graph['edges'][:3]) or 'None'}"
        else:
            scene_info = f"Scene contains {len(scene_graph['nodes'])} objects."
        
        input_text = f"Question: {question}\n{scene_info}\nAnswer:"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=150,
                num_beams=6,
                early_stopping=True
            )
        answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        
        # Handle invalid or unexpected outputs
        if "not_duplicate" in answer.lower() or "not_entailment" in answer.lower() or "nodes" in answer.lower():
            answer = "Unknown response"
        
        return answer

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    vlm = NavigationVLM("config.yaml")
    vlm.train(vlm.config['fine_tune_output'])
    test_image = "data/ai2thor_0_0_1.png"
    if os.path.exists(test_image):
        answer = vlm.query(test_image, "Is the table to the right of the chair?")
        print("Answer:", answer)

if __name__ == "__main__":
    main()