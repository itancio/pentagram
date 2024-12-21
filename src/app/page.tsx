"use server";

import ImageGenerator from "@/components/ImageGenerator";
import { generateImage } from "./actions/generateImage";
import Dashboard from "@/components/Dashboard";

export default async function Home() {
  return (
    <>
      <Dashboard />
    </>
  );
}
