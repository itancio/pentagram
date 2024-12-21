"use client";

import Image from "next/image";
import { useState } from "react";

interface ImageGeneratorProps {
  generateImage: (
    text: string
  ) => Promise<{ success: boolean; imageUrl?: string; error?: string }>;
}

export default function ImageGenerator({ generateImage }: ImageGeneratorProps) {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setImageUrl(null);
    setError(null);

    try {
      const result = await generateImage(inputText);

      if (!result.success || !result.imageUrl) {
        throw new Error(result.error || "Failed to generate image");
      }

      setImageUrl(result.imageUrl);
      setInputText("");
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to generate the image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between p-8">
      <main className="flex-1 flex flex-col items-center justify-center">
        {error && (
          <div className="mb-4 text-red-600 dark:text-red-400 p-50">
            {error}
          </div>
        )}

        {imageUrl && (
          <div className="w-full max-w-xl rounded-lg overflow-hidden shadow-lg">
            <Image
              src={imageUrl}
              width={1024}
              height={1024}
              alt={`Generated Image for: ${inputText}`}
              className="object-contain w-full h-auto"
            />
          </div>
        )}
      </main>

      <footer className="w-full max-w-3xl mx-auto">
        <form onSubmit={handleSubmit} className="w-full">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              className="flex-1 p-3 rounded-lg bg-black/[.05] dark:bg-white/[.06] border border-black/[.08] dark:border-white/[.145] focus:outline-none focus:ring-2 focus:ring-black dark:focus:ring-white"
              placeholder="Describe the image you want to generate..."
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!inputText || isLoading}
              className="px-6 py-3 rounded-lg bg-foreground text-background hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors disabled:opacity-50"
            >
              {isLoading ? "Generating..." : "Generate"}
            </button>
          </div>
        </form>
      </footer>
    </div>
  );
}
