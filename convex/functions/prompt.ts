import { QueryCtx } from "../_generated/server";
import { authenticatedQuery } from "./helpers";
import { v } from "convex/values";

// Retrieves all prompts associated with the current user
export const list = authenticatedQuery({
    handler: async (ctx: QueryCtx) => {
      try {
        const prompts = await ctx.db
          .query("prompts")
          .withIndex("by_user", (q) => q.eq("user", ctx.user._id))
          .filter((prompt) => !prompt.deleted) // Exclude soft-deleted prompts
          .collect();
  
        // Return the filtered prompts
        return prompts.map( (prompt) => prompt.content)
      } catch (error) {
        console.error("Error fetching prompts:", error);
        throw new Error("Failed to retrieve prompts for the user.");
      }
    },
  });
  