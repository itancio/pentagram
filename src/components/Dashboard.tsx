"use client";
import { generateImage } from "@/app/actions/generateImage";
import { AppSidebar } from "@/components/AppSidebar";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import Image from "next/image";
import { useState } from "react";

export default function Dashboard() {
  const [inputText, setInputText] = useState("");
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setImageUrl(null);
    setError(null);

    const start = Date.now();

    try {
      const result = await generateImage(inputText);

      if (!result.success || !result.imageUrl) {
        throw new Error(result.error || "Failed to generate image");
      }

      setImageUrl(result.imageUrl);
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to generate the image. Please try again.");
    } finally {
      setIsLoading(false);
      setDescription(inputText);
      setInputText("");

      setTimeout(() => {
        const endTime = Date.now();
        const elapsedTime = (endTime - start) / 1000;
        setTime(elapsedTime);
      }, 1000);
    }
  };

  return (
    <>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger className="-ml-1" />

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
            <div className="w-full max-w-3xl mx-auto"></div>
          </header>

          <main className="flex-1 flex flex-col items-center justify-center">
            {error && (
              <div className="mb-4 text-red-600 dark:text-red-400 p-50">
                {error}
              </div>
            )}
            {imageUrl && (
              <>
                <div className="w-full max-w-xl rounded-lg overflow-hidden shadow-lg">
                  <Image
                    src={imageUrl}
                    width={1024}
                    height={1024}
                    alt={`Generated Image for: ${inputText}`}
                    className="object-contain w-full h-auto"
                  />
                </div>
                <div className="p-5">
                  {time !== null
                    ? `Generated Image for "${description}" in ${time.toFixed(2)} seconds`
                    : `Generated Image for "${description}"`}
                </div>
              </>
            )}
          </main>
          {/* <Masonry cards={cards} /> */}
        </SidebarInset>
      </SidebarProvider>
    </>
  );
}
