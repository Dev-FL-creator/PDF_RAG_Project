import { useState, useEffect } from "react";

export function useSettings() {
  const [apiBase, setApiBase] = useState(() => {
    return localStorage.getItem("apiBase") || "http://localhost:8000";
  });

  useEffect(() => {
    localStorage.setItem("apiBase", apiBase);
  }, [apiBase]);

  return {
    apiBase,
    setApiBase,
    headers: {}
  };
}
