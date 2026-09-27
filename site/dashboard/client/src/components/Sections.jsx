// Transforma um texto longo em seções recolhíveis: cada <h2> vira um item clicável.
// Mantém todo o conteúdo (importante nas páginas legais), só esconde o que não foi aberto.
import { Children, isValidElement, useEffect, useRef } from 'react';

export default function Sections({ children, openFirst = true }) {
  const ref = useRef(null);
  const groups = [];
  let current = null;
  Children.forEach(children, (child) => {
    if (isValidElement(child) && child.type === 'h2') {
      current = { id: child.props.id, title: child.props.children, body: [] };
      groups.push(current);
    } else if (current) {
      current.body.push(child);
    } else {
      groups.push({ intro: true, body: [child] });
    }
  });

  // abre a seção quando o endereço aponta para ela (ex.: /termos#envio)
  useEffect(() => {
    const open = () => {
      const id = decodeURIComponent(window.location.hash.split('#').pop() || '');
      const el = id && ref.current?.querySelector(`details[data-id="${CSS.escape(id)}"]`);
      if (el) {
        el.open = true;
        el.scrollIntoView({ block: 'start' });
      }
    };
    open();
    window.addEventListener('hashchange', open);
    return () => window.removeEventListener('hashchange', open);
  }, []);

  let n = 0;
  return (
    <div className="sections" ref={ref}>
      {groups.map((g, i) =>
        g.intro ? (
          <div key={i} className="prose">
            {g.body}
          </div>
        ) : (
          <details key={i} className="section" data-id={g.id} id={g.id} open={openFirst && n++ === 0}>
            <summary>
              <h2>{g.title}</h2>
            </summary>
            <div className="prose">{g.body}</div>
          </details>
        ),
      )}
    </div>
  );
}
