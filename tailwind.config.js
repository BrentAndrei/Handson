/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      keyframes: {
        "soft-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "mesh-pulse": {
          "0%, 100%": { opacity: "0.15", transform: "scale(1)" },
          "50%": { opacity: "0.25", transform: "scale(1.05)" },
        },
      },
      animation: {
        "soft-in": "soft-in 220ms ease-out",
        "mesh-pulse": "mesh-pulse 8s ease-in-out infinite",
      },
    },
  },
  plugins: [
    require("daisyui"),
  ],
  darkMode: "class",
};