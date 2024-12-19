# Pentagram: Instagram, but with AI Images

## Features:

## Tech Stack
* Modal
* Python
* Typescript
* Firebase/ Convex

**SOURCES**:

- [Getting started with Modal](https://modal.com/docs/examples/hello_world)
- [Building an Image Generation Pipeline on Modal](https://www.youtube.com/watch?v=sHSKArbiKmU)
- [Run Stable Diffusion as a CLI, API and webUI](https://modal.com/docs/examples/text_to_image)
- [Midjourney Examples](https://www.midjourney.com/explore?tab=top)
- [NVIDIA GPU comparison](https://www.digitalocean.com/community/tutorials/h100_vs_other_gpus_choosing_the_right_gpu_for_your_machine_learning_workload)
- [Modal Playground](https://modal.com/playground/get_started)
- [Modal cold Start Guide](https://modal.com/docs/guide/cold-start)
- [Image Generation Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Modal Web endpoints](https://modal.com/docs/guide/webhooks)

## Getting Started

First, clone the GitHub repository:

```bash
git clone https://github.com/team-headstart/pentagram.git
```

Then, navigate to the project directory:

```bash
cd pentagram
```

Then, install the dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Tasks

- Take a look at the TODOs in the repo, namely:

  - `src/app/page.tsx`: This is where the user can input their prompt and generate an image. Make sure to update the UI and handle the API response to display the images generated

  - `src/app/api/generate-image/route.ts`: This is where the image generation API is implemented. Make sure to call your image generation API from Modal here

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## Common Issues & Troubleshooting
