import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import Overview from './pages/Overview.jsx';
import CategoryPage from './pages/CategoryPage.jsx';
import Classify from './pages/Classify.jsx';
import ModelCard from './pages/ModelCard.jsx';
import Terms from './pages/Terms.jsx';
import Privacy from './pages/Privacy.jsx';
import Credits from './pages/Credits.jsx';
import Compliance from './pages/Compliance.jsx';
import NotFound from './pages/NotFound.jsx';
import Chat from './pages/Chat.jsx';
import ComingSoon from './pages/ComingSoon.jsx';
import PlanetsPage from './pages/PlanetsPage.jsx';
import { IconStar } from './components/Icons.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="galaxias" element={<CategoryPage categoryId="galaxias" />} />
        <Route path="galaxias/:tipo" element={<CategoryPage key="g" categoryId="galaxias" />} />
        <Route path="nebulosas" element={<CategoryPage categoryId="nebulosas" />} />
        <Route path="nebulosas/:tipo" element={<CategoryPage key="n" categoryId="nebulosas" />} />
        <Route path="planetas" element={<PlanetsPage />} />
        <Route path="estrelas" element={<ComingSoon key="estrelas" titleKey="nav.stars" icon={IconStar} />} />
        <Route path="classificar" element={<Classify />} />
        <Route path="modelo" element={<ModelCard />} />
        <Route path="chat" element={<Chat />} />
        <Route path="termos" element={<Terms />} />
        <Route path="privacidade" element={<Privacy />} />
        <Route path="creditos" element={<Credits />} />
        <Route path="conformidade" element={<Compliance />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
