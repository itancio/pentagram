import { QueryCtx } from "../_generated/server";
import { authenticatedQuery } from "./helpers";

// Retrieves all images associated with the prompt
export const list = authenticatedQuery({
  handler: async (ctx: QueryCtx) => {
    try {
      // Fetch all images associated with the given prompt
      const images = await ctx.db
        .query("images")
        .withIndex("by_prompt", q => q.eq("prompt", ctx.prompt._id))
        .collect();

      // Get URLs for each image from storage
      const imageUrls = await Promise.all(
        images.map(image =>
          image.loc ? ctx.storage.getUrl(image.loc) : undefined
        )
      );

      // Return URLs, filtering out any undefined entries
      return imageUrls.filter(url => url !== undefined);
    } catch (error) {
      console.error("Error fetching images:", error);
      throw new Error("Failed to retrieve images for the prompt.");
    }
  },
});
