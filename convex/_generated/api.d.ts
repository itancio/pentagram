/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";
import type * as functions_helpers from "../functions/helpers.js";
import type * as functions_images from "../functions/images.js";
import type * as functions_prompt from "../functions/prompt.js";
import type * as functions_storage from "../functions/storage.js";
import type * as functions_user from "../functions/user.js";
import type * as http from "../http.js";

/**
 * A utility for referencing Convex functions in your app's API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
declare const fullApi: ApiFromModules<{
  "functions/helpers": typeof functions_helpers;
  "functions/images": typeof functions_images;
  "functions/prompt": typeof functions_prompt;
  "functions/storage": typeof functions_storage;
  "functions/user": typeof functions_user;
  http: typeof http;
}>;
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;
