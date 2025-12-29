from .wan_dataset import build_wan_dataloader
from torchtitan.config import JobConfig

def main(args):
    job_config = JobConfig()
    job_config.training.dataset = "vidgen-1m"
    job_config.training.local_batch_size = 2

    dataloader = build_wan_dataloader(1, 0, None, job_config)
    pass

if __name__ == "__main__":
    print("main starts", flush=True)

    args = None
    main(args)

    print("main ends", flush=True)