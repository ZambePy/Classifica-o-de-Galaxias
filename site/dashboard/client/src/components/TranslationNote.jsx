import { IconInfo } from './Icons.jsx';

/** Aviso exibido nas versões em inglês dos documentos legais. */
export default function TranslationNote() {
  return (
    <div className="notice">
      <IconInfo />
      <span>
        This English version is a courtesy translation. The service is governed by Brazilian law and, in case of any
        conflict, the Portuguese version prevails.
      </span>
    </div>
  );
}
