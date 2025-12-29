
import os
import sys
import torch
import unittest
import logging
import types
from os.path import abspath, dirname, join
from unittest.mock import MagicMock, patch

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT_PATH = dirname(dirname(abspath(__file__)))
print(ROOT_PATH)
# OMNIFLOW_PATH = "/root/zirui/code/OmniFlow/src"
# TORCHTITAN_PATH = "/root/zirui/code/OmniFlow/third_party/torchtitan"
OMNIFLOW_PATH = join(ROOT_PATH, "src")
TORCHTITAN_PATH = join(ROOT_PATH, "third_party", "torchtitan")

sys.path.append(OMNIFLOW_PATH)
sys.path.append(TORCHTITAN_PATH)
sys.path.append(os.path.join(OMNIFLOW_PATH, "omniflow"))

from omniflow.models.wan.configuration_wanvideo import WanVideoConfig
from omniflow.models.wan.modeling_wanvideo import WanVideoForConditionalGeneration
from omniflow.training.trainer import WanVideoTrainer
from transformers import TrainingArguments

# Mock wan_dataset to prevent slow initialization during import
sys.modules["torchtitan.experiments.wan.wan_dataset"] = MagicMock()

from torchtitan.experiments.wan.wan_model import WanVideoModel
from torchtitan.experiments.wan.wan_args import WanModelArgs
from torchtitan.experiments.wan.loss import WanLoss

# Disable Flash Attn (for deterministic testing)
import omniflow.models.wan.wan_video_dit
omniflow.models.wan.wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
try:
    import torchtitan.experiments.wan.model.wan_video_dit
    torchtitan.experiments.wan.model.wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
except ImportError:
    pass

# Constants
DEVICE = "cuda"
# Using bfloat16 as verified in verify_torchtitan_full.py
DTYPE = torch.bfloat16 
SEED = 42

class TestTorchtitanOmniFlowAlignment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Patch WanTextEncoder to be lightweight/mocked to avoid T5-XXL initialization
        cls.text_encoder_patcher = patch("omniflow.models.wan.modeling_wanvideo.WanTextEncoder")
        cls.mock_text_encoder = cls.text_encoder_patcher.start()
        cls.addClassCleanup(cls.text_encoder_patcher.stop)
        
        # Ensure seed is set before any model initialization
        torch.manual_seed(SEED)
        
        # 1. Tiny Config (Matches verify_torchtitan_full.py)
        cls.config = WanVideoConfig(
            dit_hidden_size=16,
            dit_in_channels=4,
            dit_out_channels=4,
            dit_intermediate_size=32,
            dit_num_layers=8,
            dit_num_heads=2,
            dit_text_dim=16,
            dit_freq_dim=64, # Important to match
            dit_patch_size=(1, 2, 2),
            dit_has_image_input=False,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=False,
        )

        # 2. Instantiate Models and Inject Mocks (No Global Monkeypatching)
        cls.tt_model, cls.mock_data = cls._create_torchtitan_model(cls.config)
        cls.omni_model = cls._create_omni_model(cls.config, cls.tt_model, cls.mock_data)

        # 3. Sync DiT Weights
        logger.info("Syncing DiT weights...")
        cls.omni_model.dit.load_state_dict(cls.tt_model.model.dit.state_dict())
        
        # Verify sync
        for p1, p2 in zip(cls.tt_model.model.dit.parameters(), cls.omni_model.dit.parameters()):
            if not torch.equal(p1, p2):
                raise ValueError("DiT weights mismatch after sync!")

    @classmethod
    def _create_torchtitan_model(cls, config):
        args = WanModelArgs()
        for k, v in config.__dict__.items():
            if hasattr(args, k):
                setattr(args, k, v)
        args.vae_checkpoint_path = None
        args.t5_checkpoint_path = None
        
        # Instantiate real model (fast because it's tiny)
        model = WanVideoModel(args).to(DEVICE).to(DTYPE).eval()
        
        # Generate Common Mock Data (Fixed)
        B = 2
        common_data = {}
        # Latents need to match VAE output channels
        common_data["latents"] = torch.randn(B, config.dit_in_channels, 5, 30, 52, dtype=DTYPE, device=DEVICE) # Shapes approx Tiny
        common_data["context"] = torch.randn(B, 120, config.dit_text_dim, dtype=DTYPE, device=DEVICE)
        common_data["noise"] = torch.randn_like(common_data["latents"])
        
        # Create Mocks
        class MockTextEncoder(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return common_data["context"]

        class MockVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = types.SimpleNamespace()
                self.model.z_dim = 4
                self.model.dim = 16
                self.upsampling_factor = 16

            def encode(self, x, **kwargs):
                 # Return deterministic latents
                 is_single_frame = False
                 if isinstance(x, list):
                      if len(x) == 1 and isinstance(x[0], torch.Tensor):
                           if x[0].ndim == 4 and x[0].shape[1] == 1:
                               is_single_frame = True
                 
                 if is_single_frame:
                     return common_data["latents"][:, :, 0:1]
                 return common_data["latents"]
        
        # Inject Mocks
        model.model.text_encoder = MockTextEncoder().to(DEVICE).to(DTYPE)
        model.model.vae = MockVAE().to(DEVICE).to(DTYPE)
        
        # Patch Noise Init
        def mock_noise_init(self, *args, **kwargs): return common_data["noise"]
        model.model.noise_initialize = types.MethodType(mock_noise_init, model.model)
        
        return model, common_data

    @classmethod
    def _create_omni_model(cls, config, tt_model, common_data):
        model = WanVideoForConditionalGeneration(config).to(DEVICE).to(DTYPE).eval()
        
        # Shared mocks (Use same instances as TT)
        model.text_encoder = tt_model.model.text_encoder
        model.vae = tt_model.model.vae
        
        # Patch Noise Init
        def mock_noise_init(self, *args, **kwargs): return common_data["noise"]
        model.noise_initialize = types.MethodType(mock_noise_init, model)
        
        return model

    def test_full_alignment(self):
        logger.info("Running Torchtitan vs OmniFlow Alignment...")
        torch.manual_seed(SEED)
        
        # Inputs (B=2, matching common_data)
        B = 2
        # Shape: (B, C, F, H, W)
        F, H, W, C = 5, 32, 32, 3
        video = torch.randn(B, C, F, H, W, device=DEVICE, dtype=DTYPE)
        ids = torch.randint(0, 100, (B, 128), device=DEVICE)
        mask = torch.ones((B, 128), device=DEVICE)
        TS_VAL = 500
        
        # Patch Timestep
        with patch("torch.randint", return_value=torch.tensor([TS_VAL])):
             
             # 1. Torchtitan Forward
             class MockWanLoss(WanLoss):
                def forward(self, noise_pred, target, timestep):
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    loss = loss * self.scheduler.training_weight(timestep)
                    return loss
             wan_loss = MockWanLoss(self.tt_model.scheduler)
             
             # Ensure precision sync
            #  self.tt_model.scheduler.timesteps = self.tt_model.scheduler.timesteps.bfloat16()
             
             # Run TT
             noise_pred, target, timestep = self.tt_model(video, input_ids=ids, attention_mask=mask)
             loss_tt = wan_loss(noise_pred, target, timestep)
             
             logger.info(f"Torchtitan Loss: {loss_tt.item()}")

             # 2. OmniFlow Forward
             training_args = TrainingArguments(output_dir="./tmp_test", report_to="none")
             trainer = WanVideoTrainer(model=self.omni_model, args=training_args)
             trainer.scheduler.set_timesteps(1000, training=True)
             
             # Match precision
            #  trainer.scheduler.timesteps = trainer.scheduler.timesteps.bfloat16()
             
             omni_inputs = {
                "video": video,
                "input_ids": ids,
                "attention_mask": mask,
                 "height": 32, "width": 32, "num_frames": 5, "seed": SEED,
                 "tiled": False, "tile_size": 16, "tile_stride": 8,
                 "vace_reference_image": None, "input_image": None, "control_video": None,
                 "reference_image": None, "camera_control_direction": None, "motion_bucket_id": None,
                 "end_image": None, "vace_mask": None, "vace_video": None, "vace_scale": None, "vace_video_mask": None
             }
             
             # Patch randint for OmniFlow
             with patch("torch.randint", return_value=torch.tensor([TS_VAL])):
                 loss_omni = trainer.compute_loss(self.omni_model, omni_inputs)
                 
             logger.info(f"OmniFlow Loss: {loss_omni.item()}")
        
        # Compare
        diff = abs(loss_tt.item() - loss_omni.item())
        logger.info(f"Loss Diff: {diff}")
        
        self.assertLess(diff, 1e-3) # Should be ~0.0
        if diff < 1e-4:
            logger.info("[SUCCESS] Alignment Verified!")

if __name__ == "__main__":
    unittest.main()
