import sys
import torch
sys.path.append("/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation")
#   EVERY model import in this file is LAZY -- done inside the branch that
#   needs it. None of these are importable from a plain repo checkout:
#   ICNet/icnet.py does `from backbone import ...` and BisNet/PIDNet do the
#   same flat-import trick, which only resolved on Colab because sys.path had
#   been extended with each model's own folder. `transformers` is absent
#   entirely in an mmseg-only env. At module scope any one of them makes
#   `import pretrained_models.models` fail, taking the mmseg arm down with it.
import torch.nn.functional as F

class Models():
  def __init__(self,config):
    self.config = config
    self.name = config.model.name
    self.device = config.experiment.device
    self.model = None

  def get(self):
    if 'pidnet' in self.config.model.name:
      from pretrained_models.PIDNet.model import PIDNet, get_pred_model
      if '_s' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/PIDNet/PIDNet_S_Cityscapes_test.pt',map_location=self.device)
      if '_m' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/PIDNet/PIDNet_M_Cityscapes_test.pt',map_location=self.device)
      if '_l' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/PIDNet/PIDNet_L_Cityscapes_test.pt',map_location=self.device)
      
  
      pidnet = get_pred_model(name = self.config.model.name, num_classes = 19).to(self.device)
      if 'state_dict' in model:
          model = model['state_dict']
      model_dict = pidnet.state_dict()
      model = {k[6:]: v for k, v in model.items() # k[6:] to start after model. in key names
                          if k[6:] in model_dict.keys()}

      pidnet.load_state_dict(model)
      self.model = pidnet
      self.model.eval()
      

    if 'bisenet' in self.config.model.name:
      from pretrained_models.BisNetV1.model import BiSeNetV1
      from pretrained_models.BisNetV2.model import BiSeNetV2
      if '_v1' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/BisNetV1/bisnetv1.pth',map_location=self.device)
        bisenet = BiSeNetV1(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      if '_v2' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/BisNetV2/bisnetv2.pth',map_location=self.device)
        bisenet = BiSeNetV2(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      self.model = bisenet
      self.model.eval()


    if 'icnet' in self.config.model.name:
      from pretrained_models.ICNet.icnet import ICNet
      model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/ICNet/Copy of resnet50_2024-12-22 08:52:50 EST-0500_176_0.661.pth.tar',map_location=self.device)
      icnet = ICNet(nclass = 19).to(self.device)
      icnet.load_state_dict(model['model_state_dict'])
      self.model = icnet
      self.model.eval()

    if 'mmseg' in self.config.model.name:
      # OUR mmseg checkpoints, not HuggingFace's. Loaders live in
      # pretrained_models/SegFormerB0 and .../SegFormerB5, one folder per
      # variant like the other models here; they return the network already
      # wrapped for the configured inference mode.
      #
      # THE INPUT CONTRACT NEEDS NO CONVERSION. BaseDataset.input_transform
      # does /255 then ImageNet mean/std, and patchreach.data.cityscapes uses
      # IMG_MEAN/IMG_STD with the SAME values, while WrappedSegModel does no
      # normalisation of its own. The tensor this repo's dataloader emits is
      # already exactly what the segmentor expects -- do NOT also call
      # norm_tensors on it, or the image is normalised twice with no error.
      #
      # 'mmseg' in model.name is what selects this branch and what makes the
      # HF branch below skip; '_b0'/'_b5' picks the variant.
      if '_b5' in self.config.model.name:
        from pretrained_models.SegFormerB5.model import get_model
      else:
        from pretrained_models.SegFormerB0.model import get_model
      self.model, info = get_model(
          self.config.model.ckpt_dir,
          device=self.device,
          slide=self.config.model.slide,
          num_classes=self.config.dataset.num_classes,
          slide_checkpoint=self.config.model.slide_checkpoint)
      print(f"[mmseg] {self.config.model.name}\n[ckpt ] {info['weights']}\n"
            f"[infer] {info['inference']}")
      self.model.eval()

    if 'segformer' in self.config.model.name and 'mmseg' not in self.config.model.name:
      from transformers import (SegformerFeatureExtractor,
                                SegformerForSemanticSegmentation)
      feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
      segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(self.device)
      self.model = segformer
      self.model.eval()


  def predict(self,image_standard,size):
    image_standard = image_standard.to(self.device)
    outputs = self.model(image_standard)
    if 'pidnet' in self.config.model.name:
      output = F.interpolate(
                    outputs[self.config.test.output_index_pidnet], size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'mmseg' in self.config.model.name:
      # WrappedSegModel returns logits [B,19,h,w] directly -- no .logits
      # attribute and no tuple indexing. Upsampled to the label size the same
      # way the other arms do it, so the loss and metric see one contract.
      output = F.interpolate(
                    outputs, size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'segformer' in self.config.model.name and 'mmseg' not in self.config.model.name:
      output = F.interpolate(
                    outputs.logits, size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'icnet' in self.config.model.name:
      output = outputs[self.config.test.output_index_icnet]

    if 'bisenet' in self.config.model.name:
      ## Images needs to be unnormalized and then normalized as:
      ## mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115]
      ## The it will give 75% miou instead of 71 and to keep things simple keeping it as it
      output = outputs[self.config.test.output_index_bisenet]

    return output

    





