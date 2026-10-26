import { Tooltip } from "./Tooltip";
import { motion } from "framer-motion";

interface Item {
  label: string;
  ready: boolean;
  error?: string | null;
  tooltip?: string;
}

interface Props {
  items: Item[];
  fps?: number | null;
  offline?: boolean;
}

/**
 * A compact status bar under the camera: one pill per subsystem (camera,
 * model) plus the live fps counter. Each pill is green when ready, amber
 * when loading, red when errored — so a teacher can tell at a glance
 * whether the pipeline is healthy without opening the console.
 *
 * `offline` is set from the browser's native `offline`/`online`
 * events (mirrored by the service worker's `offline-change` custom
 * event in main.tsx) so the banner appears automatically when the
 * network drops (the app still works from cache, since the model and WASM are cached).
 */
export function StatusBar({ items, fps, offline }: Props) {
  return (
    <>
      {offline && <motion.div className="offline-banner" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.25 }}>Offline — running from cache</motion.div>}
      <motion.div
        className="topbar"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 px-4 py-2.5 sm:px-6">
          <motion.div
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.05 }}
          >
            <span className="size-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.6)]" aria-hidden="true" />
            <span className="text-sm font-bold tracking-tight text-[var(--ink)]">HandSon</span>
            <span className="hidden text-xs font-medium text-[var(--ink-soft)] sm:inline">· Filipino Sign Language</span>
          </motion.div>
          <motion.div
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.05 }}
          >
            {items.map((it, i) => (
              <motion.span
                key={it.label}
                className={`glass-pill ${
                  it.error
                    ? "border-coral bg-coral/10 text-coral"
                    : it.ready
                    ? "border-lime bg-lime/10 text-lime-deep"
                    : "border-sun bg-sun/10 text-sun-deep"
                }`}
                title={it.error ?? undefined}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.2, delay: 0.1 + i * 0.05 }}
              >
                <span
                  className={`size-1.5 rounded-full ${
                    it.error
                      ? "bg-coral"
                      : it.ready
                      ? "bg-lime"
                      : "animate-pulse bg-sun"
                  }`}
                  aria-hidden="true"
                />
                {it.tooltip ? (
                  <Tooltip text={it.tooltip}>{it.label}</Tooltip>
                ) : (
                  it.label
                )}
              </motion.span>
            ))}
            {fps ? <motion.span className="glass-pill text-[var(--ink)]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3, delay: 0.2 }}>{fps} fps</motion.span> : null}
          </motion.div>
        </div>
      </motion.div>
    </>
  );
}
