import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // Users Table
  users: defineTable({
    username: v.string(),
    avatar: v.string(),
    clerkId: v.string(), // External ID from Clerk for authentication
  })
    .index("by_clerk_id", ["clerkId"]) // Index to efficiently fetch users by Clerk ID
    .index("by_username", ["username"]), // Index for username uniqueness and queries

  // Prompts Table
  prompts: defineTable({
    user: v.id("users"), // References the 'users' table
    content: v.string(), // Prompt content entered by the user
    deleted: v.optional(v.boolean()), // Soft deletion flag
  }).index("by_user", ["user"]), // Index to fetch prompts for a user

  // Images Table
  images: defineTable({
    prompt: v.id("prompts"), // References the 'prompts' table
    loc: v.optional(v.id("storage")), // Reference to Convex storage for the actual image
  }).index("by_prompt", ["prompt"]), // Index to fetch
});
