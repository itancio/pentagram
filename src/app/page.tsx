"use client";

import Dashboard from "@/components/Dashboard";

import { useMutation, useQuery } from "convex/react";
import { useState } from "react";
import { api } from "../../convex/_generated/api";

export default function Home() {
  // const [inputText, setInputText] = useState<string>("");
  // const [isLoading, setIsLoading] = useState<boolean>(false);

  // const handleSubmit = async (e: React.FormEvent) => {
  //   e.preventDefault();
  //   setIsLoading(true);

  //   try {
  //     const response = await fetch("/api/generate-image", {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify({ text: inputText }),
  //     });

  //     const data = await response.json();
  //     console.log(data);
  //     setInputText("");
  //   } catch (error) {
  //     console.error("Error:", error);
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  const prompts = useQuery(api.functions.prompt.list);
  const createPrompt = useMutation(api.functions.prompt.create);
  const [input, setInput] = useState("");

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    createPrompt({ sender: "Alice", content: input });
    setInput("");
  };

  return (
    <>
      {/* Messages */}
      <div>
        {prompts?.map((prompt, index) => (
          <div key={index}>
            <strong>{prompt}</strong>
          </div>
        ))}
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            name="prompt"
            id="prompt"
            value={input}
            onChange={e => setInput(e.target.value)}
          />
          <button type="submit">Send</button>
        </form>
      </div>
      <Dashboard />
    </>
    // <div className="min-h-screen flex flex-col justify-between p-8">
    //   {/* Header Area */}
    //   <div className="w-full max-w-3xl mx-auto">
    //     <form onSubmit={handleSubmit} className="w-full">
    //       <div className="flex gap-2">
    //         <input
    //           type="text"
    //           value={inputText}
    //           onChange={e => setInputText(e.target.value)}
    //           className="flex-1 p-3 rounded-lg bg-black/[.05] dark:bg-white/[.06] border border-black/[.08] dark:border-white/[.145] focus:outline-none focus:ring-2 focus:ring-black dark:focus:ring-white"
    //           placeholder="Describe the image you want to generate..."
    //           disabled={isLoading}
    //         />
    //         <button
    //           type="submit"
    //           disabled={isLoading}
    //           className="px-6 py-3 rounded-lg bg-foreground text-background hover:bg-[#383838] dark:hover:bg-[#ccc] transition-colors disabled:opacity-50"
    //         >
    //           {isLoading ? "Generating..." : "Generate"}
    //         </button>
    //       </div>
    //     </form>
    //   </div>

    //   {/* Dashboard */}
    //   <Dashboard />
    // </div>
  );
}
