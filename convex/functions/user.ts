import {
  internalMutation,
  MutationCtx,
  query,
  QueryCtx,
} from "../_generated/server";
import { v } from "convex/values";

export const get = query({
  handler: async ctx => {
    return await getCurrentUser(ctx);
  },
});

export const upsert = internalMutation({
  args: {
    username: v.string(),
    avatar: v.string(),
    clerkId: v.string(),
  },
  handler: async (ctx, { username, avatar, clerkId }) => {
    try {
      // Fetch the user based on clerkId
      const user = await getUserByClerkId(ctx, clerkId);

      if (user) {
        // Update existing user
        await ctx.db.patch(user._id, {
          username,
          avatar,
        });
        console.log(`User updated: ${username}`);

        // Return the updated user
        return { status: "updated", user: { ...user, username, avatar } };
      } else {
        // Insert a new user
        const newUserId = await ctx.db.insert("users", {
          username,
          avatar,
          clerkId,
        });

        // Retrieve the newly created user
        const newUser = await ctx.db.get(newUserId);

        console.log(`New user created: ${username}`);

        // Return the new user
        return { status: "created", user: newUser };
      }
    } catch (error) {
      console.error("Error in upsert:", error);
      throw new Error("Failed to upsert user.");
    }
  },
});

export const getCurrentUser = async (ctx: QueryCtx | MutationCtx) => {
  const identity = await ctx.auth.getUserIdentity();
  console.log("User identity retrieved:", identity?.subject);

  if (!identity) {
    console.log("No user identity found.");
    return null;
  }

  const user = await getUserByClerkId(ctx, identity.subject);
  console.log("Fetched user:", user);
  return user;
};

// Return a unique clerkID
const getUserByClerkId = async (
  ctx: QueryCtx | MutationCtx,
  clerkId: string
) => {
  const user = await ctx.db
    .query("users")
    .withIndex("by_clerk_id", q => q.eq("clerkId", clerkId))
    .unique();
  console.log("ClerkId: ", user?.username);
  return user;
};
