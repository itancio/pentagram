import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# MODEL_ID = "stabilityai/sdxl-turbo"
BATCH_SIZE = 1
NUM_INFERENCE_STEPS = 10

def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]",
    "transformers",
    "accelerate",
    "diffusers",
    'requests',
).run_function(download_model)

app = modal.App("pentagram", image=image)


@app.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,     # 5 minutes
    secrets=[modal.Secret.from_name("custom-secret"),
             modal.Secret.from_name("huggingface-secret")],
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
        "/root/.triton": modal.Volume.from_name(
            "triton-cache", create_if_missing=True
        ),
        "/root/.inductor-cache": modal.Volume.from_name(
            "inductor-cache", create_if_missing=True
        ),
    }
)
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        from transformers.utils import move_cache
        import torch

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        move_cache()
    
        # Add memory efficiency stategy
        self.pipeline.to("cuda")

        self.enhance_and_optimize()

        self.HF_TOKEN = os.environ['HF_TOKEN']
        self.API_KEY = os.environ['API_KEY']

    
    # TODO: optimizer
    def enhance_and_optimize(self):
        import torch
        from diffusers import EulerDiscreteScheduler

        self.set_torch_config()

        # Add scheduler
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)

        # change UNet and VAE's memory layout to channel's last when compiling to ensure max speed
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

        # Compile the UNet and VAE.
        self.pipeline.unet = torch.compile(self.pipeline.unet, mode="max-autotune", fullgraph=True)
        self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="max-autotune", fullgraph=True)

        print("finished pipeline optimization")


    def set_torch_config(self):
        import torch
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True


    @modal.web_endpoint()
    def generate(
        self, 
        request: Request, 
        prompt: str = Query(..., description="The prompt for image generation")):
        
        import torch

        # Security check for authorized access to our API
        api_key = request.headers.get("X-API-Key")

        print("api_key received: ", api_key)
        print("api_key in secret:", self.API_KEY)

        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized Access. Must have valid API key"
            )

        image = self.pipeline(
            prompt = prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=3.5,
        ).images[0]

        # store to memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    
    @modal.web_endpoint()
    def health_check(self, request: Request):
        """Lightweight endpoint for keeping the container warm"""
        return {
            "status": "ok", 
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("custom-secret")]
)  # keep warm for every 5 minutes
def keep_warm():
    health_url = "https://irvin-tanc--pentagram-model-health-check.modal.run"
    generate_url = "https://irvin-tanc--pentagram-model-generate.modal.run"

    # First check endpoint(no API key needed)
    try:
        health_response = requests.get(health_url)
        health_response.raise_for_status()  # Raise an error for HTTP codes >= 400

        health_response = health_response.json()
        status = health_response["status"]
        timestamp = health_response["timestamp"]
        print(f"Health check endpoint tested {status} at {timestamp}")
    except requests.RequestException as e:
        print(f"Health check failed: {e}")


    # Mkae a test request to generate endpoint with API key
    headers = {"X-API-Key": os.environ["API_KEY"]}
    requests.get(generate_url, headers=headers) 
    print(f"Generate endpoint tested successfully at {datetime.now(timezone.utc).isoformat()}")


################################################################################################


