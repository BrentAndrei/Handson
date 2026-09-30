import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge conditional class names with Tailwind conflict resolution.
 * The project already ships clsx + tailwind-merge, so this is the same helper
 * the shadcn ecosystem uses — no new dependency.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}