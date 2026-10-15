import React from "react";

// Verification badge: ✓ Supported / ⚠ Partially Supported / ✕ Unsupported, per the spec.
const STYLES = {
  SUPPORTED: { label: "Supported", icon: "✓", classes: "bg-emerald-100 text-emerald-800 border-emerald-300" },
  PARTIALLY_SUPPORTED: { label: "Partially Supported", icon: "⚠", classes: "bg-amber-100 text-amber-800 border-amber-300" },
  UNSUPPORTED: { label: "Unsupported", icon: "✕", classes: "bg-rose-100 text-rose-800 border-rose-300" },
};

export default function VerdictBadge({ verdict, size = "sm" }) {
  const style = STYLES[verdict] || STYLES.UNSUPPORTED;
  const sizeClasses = size === "sm" ? "text-xs px-2 py-0.5" : "text-sm px-2.5 py-1";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border font-medium ${style.classes} ${sizeClasses}`}>
      <span aria-hidden="true">{style.icon}</span>
      {style.label}
    </span>
  );
}
