# Ulcer Staging Guide

## Medical Background on Ulcer Staging

Foot ulcers are classified into stages based on their depth, severity, and presence of complications. The staging system helps healthcare professionals determine the appropriate treatment protocol. Our system uses a simplified Wagner classification system adapted for automated detection:

### Stage 1: Superficial Ulcer
- **Description**: Partial or full-thickness wound confined to the skin surface
- **Characteristics**: Epidermis and possibly dermis affected, no penetration to deeper tissues
- **Appearance**: Shallow wound, may appear as redness or open sore without deep penetration

### Stage 2: Deep Ulcer
- **Description**: Ulcer extends through the dermis into subcutaneous tissue
- **Characteristics**: Exposes underlying structures like tendons, joint capsules, or bone
- **Appearance**: Deeper wound with visible underlying tissue structures

### Stage 3: Infected Ulcer
- **Description**: Deep ulcer with signs of infection, abscess formation, or osteomyelitis
- **Characteristics**: Presence of pus, inflammation, swelling, warmth, or systemic infection indicators
- **Appearance**: Deep wound with infected appearance, possible drainage or abscess

### Stage 4: Gangrene/Final Stage
- **Description**: Tissue necrosis with gangrenous changes
- **Characteristics**: Blackened, dead tissue indicating critical ischemia or severe infection
- **Appearance**: Black or dark-colored necrotic tissue, extensive tissue death

## Annotation Guidelines for Dataset Preparation

### Image Requirements
- Clear, well-lit images of foot ulcers
- High resolution (minimum 224x224 pixels recommended)
- Minimal occlusions or artifacts
- Consistent lighting conditions

### Staging Criteria

#### Stage 1:
- Wound confined to superficial layers
- No visible deeper tissue involvement
- May show redness or minor skin breaks
- Depth appears minimal

#### Stage 2:
- Visible deeper tissue involvement
- Tendon, bone, or joint capsule potentially visible
- More pronounced depth compared to Stage 1
- Subcutaneous tissue exposed

#### Stage 3:
- Signs of active infection present
- Pus, discharge, or inflammatory response
- Possible abscess formation
- Redness extending beyond wound edges

#### Stage 4:
- Black, necrotic tissue clearly visible
- Extensive tissue death
- Dark coloration indicating gangrene
- Critical perfusion issues evident

### Annotation Process
1. Examine the ulcer image carefully
2. Assess depth and tissue involvement
3. Check for signs of infection or necrosis
4. Assign the appropriate stage based on criteria
5. Verify annotation with clinical knowledge

## Model Training Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- ultralytics
- scikit-learn
- numpy
- pillow
- yaml

### Directory Structure
```
dataset_staged/
├── train/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── stage4/
└── val/
    ├── stage1/
    ├── stage2/
    ├── stage3/
    └── stage4/
```

### Training the Stage Classifier
```bash
python train_stage_classifier.py --config stage_config.yaml --train_dir dataset_staged/train --val_dir dataset_staged/val
```

### Configuration Options
- Model type: EfficientNet-B0 (default), ResNet50, or MobileNetV2
- Input size: 224x224 (default)
- Number of classes: 4 (Stage 1-4)
- Learning rate: 0.0001 (default)
- Batch size: 32 (default)

## Troubleshooting Common Issues

### Model Performance Issues
- **Low accuracy**: Check if dataset is balanced across all 4 stages
- **Overfitting**: Implement stronger regularization or data augmentation
- **Poor generalization**: Increase validation set diversity

### Dataset Issues
- **Class imbalance**: Apply class weighting during training
- **Insufficient samples**: Collect more images for underrepresented stages
- **Poor quality images**: Implement image preprocessing filters

### Integration Issues
- **Model loading errors**: Verify model path in stage_config.yaml
- **Memory issues**: Reduce batch size during inference
- **Slow inference**: Consider using a lighter model architecture

### Common Error Messages
- `"Stage classifier not available"`: Model file path is incorrect or model hasn't been trained yet
- `"CUDA out of memory"`: Reduce batch size or use CPU-only inference
- `"Invalid image format"`: Check input image format and preprocessing steps

## Performance Expectations
- **Target accuracy**: ≥85% for stage classification
- **Inference time**: <2 seconds per image
- **Detection preservation**: Original detection accuracy should remain unchanged
- **Success rate**: ≥95% of images processed successfully