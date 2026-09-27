import { useState, useRef, useEffect, useMemo, useCallback } from "react";

interface AutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  vocabulary: string[];
  disabled?: boolean;
  placeholder?: string;
}

export function Autocomplete({
  value,
  onChange,
  onSubmit,
  vocabulary,
  disabled,
  placeholder,
}: AutocompleteProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const query = value.trim().toLowerCase();

  const suggestions = useMemo(() => {
    if (!query) return vocabulary.slice().sort();
    const filtered = vocabulary.filter((word) => word.toLowerCase().startsWith(query));
    return filtered.sort((a, b) => a.localeCompare(b));
  }, [query, vocabulary]);

  useEffect(() => {
    setActiveIndex(-1);
  }, [query]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (activeIndex >= 0 && listRef.current) {
      const item = listRef.current.children[activeIndex] as HTMLElement;
      if (item) item.scrollIntoView({ block: "nearest" });
    }
  }, [activeIndex]);

  const handleSelect = useCallback(
    (word: string) => {
      onChange(word);
      setIsOpen(false);
      setActiveIndex(-1);
    },
    [onChange]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!isOpen || suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      handleSelect(suggestions[activeIndex]);
    } else if (e.key === "Escape") {
      setIsOpen(false);
      setActiveIndex(-1);
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <textarea
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          if (e.target.value.trim()) setIsOpen(true);
        }}
        onFocus={() => {
          if (!value.trim()) setIsOpen(true);
        }}
        onKeyDown={(e) => {
          handleKeyDown(e);
          if (e.key === "Enter" && !e.shiftKey && !disabled && value.trim().length > 0 && activeIndex < 0) {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder={placeholder || "Type a sentence in English, e.g. How are you? (Enter to sign)"}
        disabled={disabled}
        rows={3}
        autoComplete="off"
        aria-autocomplete="list"
        aria-controls={isOpen ? "autocomplete-list" : undefined}
        aria-activedescendant={activeIndex >= 0 ? `suggestion-${activeIndex}` : undefined}
        className="min-h-[4.5rem] w-full flex-1 resize-y rounded-xl border border-cyan-400/20 bg-white/5 px-4 py-3 text-base text-white outline-none transition placeholder:text-slate-500 focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
      />
      {isOpen && suggestions.length > 0 && (
        <ul
          id="autocomplete-list"
          ref={listRef}
          className="absolute z-20 mt-2 w-full max-h-60 overflow-auto rounded-xl border border-slate-700/50 bg-slate-900/95 backdrop-blur-xl shadow-2xl"
        >
          {suggestions.map((word, i) => (
            <li
              key={word}
              id={`suggestion-${i}`}
              onClick={() => handleSelect(word)}
              className={`cursor-pointer px-4 py-2.5 text-sm transition-colors duration-150 ${
                i === activeIndex
                  ? "bg-cyan-400/15 text-cyan-300"
                  : "text-slate-300 hover:bg-white/5"
              }`}
              role="option"
              aria-selected={i === activeIndex}
            >
              {word}
            </li>
          ))}
        </ul>
      )}
      {isOpen && value.trim() && suggestions.length === 0 && (
        <div className="absolute z-20 mt-2 w-full rounded-xl border border-slate-700/50 bg-slate-900/95 backdrop-blur-xl px-4 py-3 text-sm text-slate-500 shadow-2xl">
          No matches found
        </div>
      )}
    </div>
  );
}