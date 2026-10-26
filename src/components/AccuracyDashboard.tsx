import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { getAccuracyLogger } from "../inference/accuracyLog";
import type { EmissionEvent, AttemptEvent } from "../inference/accuracyLog";

interface Props {
  show: boolean;
}

function useEvents(): AccuracyEvent[] {
  const [events, setEvents] = useState(() => getAccuracyLogger().getEvents());
  useEffect(() => {
    const id = setInterval(() => setEvents(getAccuracyLogger().getEvents()), 500);
    return () => clearInterval(id);
  }, []);
  return events;
}

type AccuracyEvent = EmissionEvent | AttemptEvent;

function EmissionsPanel({ emissions }: { emissions: EmissionEvent[] }) {
  const stats = getAccuracyLogger().emissionStats();
  const switches = getAccuracyLogger().confusionSwitches();

  const recent = emissions.slice(-5).reverse();

  return (
    <div className="space-y-3">
      <div className="flex justify-between text-sm font-semibold text-ink-soft">
        <span>Emissions</span>
        <span>{emissions.length}</span>
      </div>

      {recent.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-ink-soft">Recent</p>
          <div className="space-y-1">
            {recent.map((e) => (
              <div key={`${e.t}-${e.label}`} className="flex items-center gap-2 text-xs">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    e.confidence >= 0.7 ? "bg-teal" : e.confidence >= 0.4 ? "bg-sun" : "bg-coral"
                  }`}
                  aria-hidden="true"
                />
                <span className="font-medium">{e.label}</span>
                <span className="text-ink-soft ml-auto">{Math.round(e.confidence * 100)}%</span>
                {e.runnerUp && (
                  <span className="text-ink-soft opacity-60">←{e.runnerUp}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {stats.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-ink-soft">Top labels</p>
          {stats.slice(0, 5).map((s) => (
            <div key={s.label} className="flex items-center gap-2 text-xs">
              <span className="w-12 text-ink-soft">{s.label.slice(0, 12)}</span>
              <span className="text-ink font-medium">{s.count}</span>
              <span className="text-ink-soft opacity-60 ml-auto">{Math.round(s.meanConfidence * 100)}% avg</span>
            </div>
          ))}
        </div>
      )}

      {switches.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-ink-soft text-coral">Confusion switches</p>
          {switches.slice(0, 3).map(({ from, to, count }) => (
            <div key={`${from}→${to}`} className="text-xs text-ink-soft">
              {from} → {to}: <span className="font-medium text-coral">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AttemptsPanel({ attempts }: { attempts: AttemptEvent[] }) {
  const stats = getAccuracyLogger().attemptStats();

  return (
    <div className="space-y-3">
      <div className="flex justify-between text-sm font-semibold text-ink-soft">
        <span>Practice attempts</span>
        <span>{attempts.length}</span>
      </div>

      {stats.length === 0 ? (
        <p className="text-xs text-ink-soft">No attempts yet — practice a sign to get started.</p>
      ) : (
        <div className="space-y-1.5">
          {stats.slice(0, 5).map((s) => {
            const hitRateColor = s.hitRate >= 0.6 ? "text-teal-deep" : s.hitRate >= 0.3 ? "text-sun" : "text-coral";
            return (
              <div key={s.target} className="flex items-center justify-between text-xs">
                <span className="w-16 truncate text-ink">{s.target.slice(0, 16)}</span>
                <span className={hitRateColor}>
                  {s.hits}/{s.attempts} hit
                </span>
                <span className="text-ink-soft">
                  {s.meanScore !== null ? `${Math.round(s.meanScore * 100)}% score` : "—"}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function AccuracyDashboard({ show }: Props) {
  const events = useEvents();
  const emissions = events.filter((e): e is EmissionEvent => e.type === "emission");
  const attempts = events.filter((e): e is AttemptEvent => e.type === "attempt");

  const totalEvents = emissions.length + attempts.length;

  return (
    <AnimatePresence>
      {show && (
        <motion.aside
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <div className="glass-card space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="card-title">Accuracy log</h3>
              <span className="text-xs font-medium text-ink-soft">{totalEvents} events</span>
            </div>

            {totalEvents === 0 ? (
              <p className="text-sm text-ink-soft">No data yet. Emissions are logged during practice.</p>
            ) : (
              <div className="space-y-5">
                <EmissionsPanel emissions={emissions} />
                <AttemptsPanel attempts={attempts} />
              </div>
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
