import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import '@fontsource/figtree/400.css';
import '@fontsource/figtree/500.css';
import '@fontsource/figtree/600.css';
import '@fontsource/figtree/700.css';
import '@fontsource/figtree/800.css';
// Syne ExtraBold: só no nome da marca (logotipo), como no manual da marca
import '@fontsource/syne/800.css';
import './styles/global.css';
import App from './App.jsx';
import { DataProvider } from './components/DataContext.jsx';
import { PrefsProvider } from './prefs.jsx';

// VITE_HASH_ROUTER=1 gera uma versão estática (sem servidor) usada só para pré-visualização.
const Router = import.meta.env.VITE_HASH_ROUTER === '1' ? HashRouter : BrowserRouter;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <PrefsProvider>
      <Router>
        <DataProvider>
          <App />
        </DataProvider>
      </Router>
    </PrefsProvider>
  </StrictMode>,
);
