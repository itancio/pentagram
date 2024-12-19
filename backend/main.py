
#================================================================================================
#
# BASIC SETUP
#
#================================================================================================

from io import BytesIO
import random
import time
from pathlib import Path

import modal

# Running Flux fast
# We’ll make use of the full CUDA toolkit in this example, so we’ll build our container image off of the nvidia/cuda base.
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b" # original sha
diffusers_commit_sha = "9c0e20de61a6e0adcec706564cee739520c1d2f4" 

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).entrypoint([])

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "numpy<2",
        "accelerate==0.33.0",
        # "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "torchvision==0.20.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).  # faster downloads
)

# Torch compilation needs to be re-executed when each new container starts, 
# So we turn on some extra caching to reduce compile times for later containers.
flux_image = flux_image.env(
    {"TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"}
).env({"TORCHINDUCTOR_FX_GRAPH_CACHE": "1"})


basic_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

# Switch between BASIC or FLUX
image = flux_image

# Creates the app. All Modal programs need an App — an object that acts as a recipe for the application. Let’s give it a friendly name.
app = modal.App("pentagram-app")

# The `image.imports()` lets us conditionally import in the global scope.
# This is needed because we might have the dependencies installed locally,
# but we know they are installed inside the custom image.

with image.imports():
    from diffusers import (FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline)
    import torch
    from fastapi import Response


#================================================================================================
#
# AVAILABLE DIFFUSION MODELS and DEFAULT PARAMS
#
#================================================================================================
  
MINUTES = 60 #seconds
VARIANT = "dev"  # "schnell" or "dev", but note [dev] requires you to accept terms and conditions on HF
NUM_INFERENCE_STEPS = 50  # use ~50 for [dev], smaller (~4) for [schnell]

bf_model = f"black-forest-labs/FLUX.1-{VARIANT}"
sd_model = "stabilityai/stable-diffusion-3.5-large-turbo"
sdx1_model = "stabilityai/sdxl-turbo"

adamo_model = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
adamo_revision_id = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

#================================================================================================
#
# MODEL CLASS
#
#================================================================================================
@app.cls(
    image=image,
    gpu="a10g",     # Cheapest GPU
    container_idle_timeout=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
    "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
    "/root/.triton": modal.Volume.from_name(
        "triton-cache", create_if_missing=True
    ),
    "/root/.inductor-cache": modal.Volume.from_name(
        "inductor-cache", create_if_missing=True
    ),
},
)
class Model:
    compile: int = (  # see section on torch.compile below for details
      modal.parameter(default=0)
    )

    def setup_model(self)
    @modal.build()
    @modal.enter()
    def __init__(self):
        """
        Initialize our diffusion model
        """
        sdx1_model = "stabilityai/sdxl-turbo"
        self.diffuser = StableDiffusion3Pipeline.from_pretrained(
            sdx1_model,
            torch_dtype=torch.bfloat16,
        )

    @modal.enter()
    def move_to_gpu(self):
        self.pipe.to("cuda")

    def generateImage(self, prompt, batch_size:int = 4):
        imageOutput = self.diffuser(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=NUM_INFERENCE_STEPS,
            max_sequence_length=512,
            num_images_per_prompt=batch_size,  # outputting multiple images per prompt is much cheaper than separate calls
            num_inference_steps=4,  # turbo is tuned to run in four steps
            guidance_scale=3.5,  # turbo doesn't use CFG
            max_sequence_length=512,  # T5-XXL text encoder supports longer sequences, more complex prompts
        ).images
        return imageOutput

    def run(
        self, prompt: str, batch_size: int = 4, seed: int = None
    ) -> list[bytes]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)

        images = self.generateImage(prompt, batch_size)

        buffer = []
        for image in images:
            with BytesIO() as buf:
                image.save(buf, format="PNG")
                buffer.append(buf.getvalue())
        torch.cuda.empty_cache()  # reduce fragmentation
        return buffer

    # Sometimes your endpoint needs to do something before it can handle its first request,
    # like get a value from a database or set the value of a variable.
    # If that step is expensive, like [loading a large ML model](https://modal.com/docs/guide/model-weights),
    # it'd be a shame to have to do it every time a request comes in!

    # Web endpoints can be methods on a [`modal.Cls`](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-functions-and-parameters).
    # Note that they don't need the [`modal.method`](https://modal.com/docs/reference/modal.method) decorator.
    @modal.web_endpoint(docs=True)
    def web(self, prompt: str, seed: int = None):
        return Response(
            content=self.run.local(  # run in the same container
                prompt, batch_size=1, seed=seed
            )[0],
            media_type="image/png",
        )


#================================================================================================
#
# MODAL APP'S ENTRYPOINT
#
# This will trigger the run locally. The first time we run this,
# it will take 1-2 min. When we run this subsequent times, the image is already built,
# and it will run much faster.
#
#================================================================================================


@app.local_entrypoint()
def entrypoint(
    samples: int = 4,
    prompt: str = "A princess riding on a pony",
    batch_size: int = 4,
    seed: int = None,
):
    print(
        f"prompt => {prompt}",
        f"samples => {samples}",
        f"batch_size => {batch_size}",
        f"seed => {seed}",
        sep="\n",
    )

    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    inference_service = Model()

    for sample_idx in range(samples):
        start = time.time()
        images = inference_service.run.remote(prompt, batch_size, seed)
        duration = time.time() - start
        print(f"Run {sample_idx+1} took {duration:.3f}s")
        if sample_idx:
            print(
                f"\tGenerated {len(images)} image(s) at {(duration)/len(images):.3f}s / image."
            )
        for batch_idx, image_bytes in enumerate(images):
            output_path = (
                output_dir
                / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            )
            if not batch_idx:
                print("Saving outputs", end="\n\t")
            print(
                output_path,
                end="\n" + ("\t" if batch_idx < len(images) - 1 else ""),
            )
            output_path.write_bytes(image_bytes)


#================================================================================================
#
# FRONT-END
#
#================================================================================================


frontend_path = Path(__file__).parent / "frontend"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jinja2==3.1.4", "fastapi[standard]==0.115.4")
    .add_local_dir(frontend_path, remote_path="/assets")
)


@app.function(
    image=web_image,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.templating import Jinja2Templates

    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Inference.web.web_url,
                "model_name": "Stable Diffusion 3.5 Large Turbo",
                "default_prompt": "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")
