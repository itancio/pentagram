import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    if (!text) {
      return NextResponse.json(
        { success: false, error: "Text parameter is missing" },
        { status: 400 }
      );
    }

    // Ensure the MODAL_URL is set
    const urlName = process.env.MODAL_URL;
    if (!urlName) {
      return NextResponse.json(
        { success: false, error: "MODAL_URL environment variable is not set" },
        { status: 500 }
      );
    }

    const url = new URL(urlName);
    url.searchParams.set("prompt", text);

    console.log("Requesting URL:", url.toString());
    console.log(process.env.API_KEY);

    // Call the Image Generation API
    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        "X-API-Key": process.env.API_KEY || "", // Ensure your API key is properly set
        Accept: "image/jpeg",
      },
    });

    console.log("response received from Modal app:", response);

    if (!response.ok) {
      const errorMessage = await response.text();
      console.error("API Error:", errorMessage);
      throw new Error(
        `HTTP error! stats: ${response.status}, message: ${errorMessage}`
      );
    }

    // Upload image to vercel
    const imgBuffer = await response.arrayBuffer();

    const filename = `${crypto.randomUUID()}.jpg`
    const blog = await put(
      filename, 
      imgBuffer, 
      {
        access: "public",
        contentType: "image/jpeg"
      })

    const responseData = response.body;

    return NextResponse.json({
      success: true,
      data: responseData,
    });
  } catch (err) {
    console.error("Error processing request:", err);
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}
