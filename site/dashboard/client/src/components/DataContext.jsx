import { createContext, useContext, useEffect, useState } from 'react';
import { api } from '../api.js';

const DataContext = createContext(null);

export function DataProvider({ children }) {
  const [stats, setStats] = useState({ total: 0, byCategory: {}, byType: {}, meanConfidence: null, lastUpdate: null });
  const [health, setHealth] = useState({ status: 'carregando', model: { connected: false } });

  useEffect(() => {
    let alive = true;
    api.stats().then((s) => alive && setStats(s));
    api.health().then((h) => alive && setHealth(h));
    return () => {
      alive = false;
    };
  }, []);

  return <DataContext.Provider value={{ stats, health }}>{children}</DataContext.Provider>;
}

export const useData = () => useContext(DataContext);
