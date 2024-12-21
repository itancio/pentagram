import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

# MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_ID = "stabilityai/sdxl-turbo"
BATCH_SIZE = 4
NUM_INFERENCE_STEPS = 8

def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True
    )

image = modal.Image.debian_slim().pip_install(
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
             modal.Secret.from_name("huggingface-secret")]
)
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    
        # Add memory efficiency stategy
        self.pipeline.to("cuda")

        self.HF_TOKEN = os.environ['HF_TOKEN']
        self.API_KEY = os.environ['API_KEY']

        # TODO: optimizer

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
            guidance_scale=0.0,
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
