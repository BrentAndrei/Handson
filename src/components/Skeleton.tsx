interface SkeletonProps {
  className?: string;
}

/**
 * A shimmering placeholder for a text line. Used inside the
 * camera skeleton while the MediaPipe landmarker + TF.js model
 * finish loading — without it the first several seconds look like a blank
 * black rectangle, which reads as broken rather than loading.
 */
export function SkeletonLine({ className = "" }: SkeletonProps) {
  return <div className={`skeleton-line skeleton ${className}`} />;
}

/**
 * Skeleton for the camera panel: a dark frame with a shimmering caption
 * line. Replaces the camera while the MediaPipe landmarker + TF.js model
 * finish loading — without it the first several seconds look like a blank
 * black rectangle, which reads as broken rather than loading.
 */
export function CameraSkeleton() {
  return (
    <div className="skeleton-camera absolute inset-0 flex flex-col items-center justify-center gap-4 p-6">
      <div className="size-12 animate-pulse rounded-full border-4 border-white/15 border-t-cyan-400" />
      <div className="w-40">
        <SkeletonLine className="h-3 bg-white/20" />
      </div>
      <div className="w-28">
        <SkeletonLine className="h-2 bg-white/15" />
      </div>
    </div>
  );
}
