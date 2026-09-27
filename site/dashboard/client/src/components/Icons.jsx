// Ícones em linha (24×24, traço), desenhados para o projeto.
const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

export const IconOverview = (p) => (
  <svg {...base} {...p}>
    <rect x="3" y="3" width="7.5" height="9" rx="2" />
    <rect x="13.5" y="3" width="7.5" height="5" rx="2" />
    <rect x="13.5" y="11" width="7.5" height="10" rx="2" />
    <rect x="3" y="15" width="7.5" height="6" rx="2" />
  </svg>
);

export const IconGalaxy = (p) => (
  <svg {...base} {...p}>
    <path d="M12 12c0-2 1.8-3.2 3.6-2.6 2.6.9 3 4.6.6 6.6-3 2.5-8.2 1.4-9.6-2.5C5 9.2 8.4 4.8 13 4.6" />
    <path d="M12 12c0 2-1.8 3.2-3.6 2.6-2.6-.9-3-4.6-.6-6.6 3-2.5 8.2-1.4 9.6 2.5 1.6 4.3-1.8 8.7-6.4 8.9" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
  </svg>
);

export const IconNebula = (p) => (
  <svg {...base} {...p}>
    <path d="M6.5 17.5c-2.2 0-3.5-1.6-3.5-3.3 0-1.9 1.6-3.4 3.5-3.3.3-2.9 2.6-5 5.4-5 2.3 0 4.2 1.4 5 3.4 2.4.1 4.1 1.9 4.1 4.1 0 2.3-1.8 4.1-4.2 4.1H6.5Z" />
    <path d="M9 13.5h.01M13 11.5h.01M15.5 14.5h.01" strokeWidth="2.6" />
  </svg>
);

export const IconUpload = (p) => (
  <svg {...base} {...p}>
    <path d="M12 16V4M7 9l5-5 5 5" />
    <path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />
  </svg>
);

export const IconModel = (p) => (
  <svg {...base} {...p}>
    <circle cx="5" cy="6" r="2" />
    <circle cx="5" cy="18" r="2" />
    <circle cx="12" cy="12" r="2" />
    <circle cx="19" cy="6" r="2" />
    <circle cx="19" cy="18" r="2" />
    <path d="M6.7 7.1l3.6 3.8M6.7 16.9l3.6-3.8M13.7 10.9l3.6-3.8M13.7 13.1l3.6 3.8" />
  </svg>
);

export const IconShield = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8.4-7.5 9.5-4.3-1.1-7.5-5-7.5-9.5V6L12 3Z" />
    <path d="M8.8 12.2l2.2 2.2 4.3-4.6" />
  </svg>
);

export const IconInfo = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" />
  </svg>
);

export const IconSpark = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3c.6 4.6 2.4 6.4 7 7-4.6.6-6.4 2.4-7 7-.6-4.6-2.4-6.4-7-7 4.6-.6 6.4-2.4 7-7Z" />
  </svg>
);

/** Ilustração do estado vazio: telescópio apontado para um ponto de interrogação estelar. */
export const EmptyIllustration = ({ accent = 'var(--sky)' }) => (
  <svg viewBox="0 0 96 96" aria-hidden="true">
    <circle cx="48" cy="48" r="44" fill="none" stroke="var(--line)" strokeDasharray="3 6" />
    <circle cx="48" cy="48" r="28" fill="none" stroke={accent} strokeOpacity=".35" />
    <path d="M40 38a8 8 0 1 1 11.5 7.2c-2.3 1.1-3.5 2.6-3.5 4.8v1.5" fill="none" stroke={accent} strokeWidth="3.5" strokeLinecap="round" />
    <circle cx="48" cy="59" r="2.4" fill={accent} />
    <path d="M78 20l1.2 3.3L82.5 24.5 79.2 25.7 78 29 76.8 25.7 73.5 24.5 76.8 23.3Z" fill="var(--mint)" />
    <circle cx="18" cy="72" r="1.8" fill="var(--lavender)" />
    <circle cx="80" cy="70" r="1.2" fill="var(--text-2)" />
  </svg>
);

export const IconSun = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" />
  </svg>
);

export const IconMoon = (p) => (
  <svg {...base} {...p}>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />
  </svg>
);

export const IconMonitor = (p) => (
  <svg {...base} {...p}>
    <rect x="3" y="4" width="18" height="12" rx="2" />
    <path d="M8 20h8M12 16v4" />
  </svg>
);

export const IconChat = (p) => (
  <svg {...base} {...p}>
    <path d="M20 12.5a7.5 7.5 0 0 1-11 6.6L4 20.5l1.4-4.4A7.5 7.5 0 1 1 20 12.5Z" />
    <path d="M9 12h.01M12.5 12h.01M16 12h.01" strokeWidth="2.6" />
  </svg>
);

export const IconSend = (p) => (
  <svg {...base} {...p}>
    <path d="M4 12 20 4l-5 16-3.5-6.5L4 12Z" />
    <path d="m11.5 13.5 3-3" />
  </svg>
);

export const IconCube = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3 20 7.5v9L12 21l-8-4.5v-9L12 3Z" />
    <path d="M4 7.5 12 12l8-4.5M12 12v9" />
  </svg>
);

export const IconPlanet = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="5.5" />
    <path d="M4.2 15.8c-1.9 1.6-2.6 3-2 3.8 1.2 1.6 7.3-.8 13.6-5.3S23 5.8 21.8 4.2c-.6-.8-2.2-.5-4.3.6" />
  </svg>
);

export const IconStar = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3.5l2.4 5.2 5.6.6-4.2 3.8 1.2 5.6L12 15.9l-5 2.8 1.2-5.6L4 9.3l5.6-.6L12 3.5Z" />
  </svg>
);
