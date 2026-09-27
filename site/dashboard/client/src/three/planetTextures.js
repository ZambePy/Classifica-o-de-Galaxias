// Texturas ILUSTRATIVAS dos planetas (equiretangulares; 2048 × 1024 no modo Leve e 4096 × 2048 no HD, para o zoom de perto), geradas por código em
// ferramentas/texturas-planetas/gerar_texturas.py. Contornos dos continentes da Terra: Natural Earth (domínio público).
import terra from '../assets/planets/terra.jpg';
import terraNuvens from '../assets/planets/terra-nuvens.jpg';
import mercurio from '../assets/planets/mercurio.jpg';
import venus from '../assets/planets/venus.jpg';
import marte from '../assets/planets/marte.jpg';
import jupiter from '../assets/planets/jupiter.jpg';
import saturno from '../assets/planets/saturno.jpg';
import urano from '../assets/planets/urano.jpg';
import netuno from '../assets/planets/netuno.jpg';
import saturnoAneis from '../assets/planets/saturno-aneis.png';
// relevo (normal maps) e brilho do oceano (rugosidade) — sessão 11
import mercurioRelevo from '../assets/planets/mercurio-relevo.jpg';
import marteRelevo from '../assets/planets/marte-relevo.jpg';
import terraRelevo from '../assets/planets/terra-relevo.jpg';
import terraRugosidade from '../assets/planets/terra-rugosidade.jpg';
import mercurioRelevoHD from '../assets/planets/4k/mercurio-relevo.jpg';
import marteRelevoHD from '../assets/planets/4k/marte-relevo.jpg';
import terraRelevoHD from '../assets/planets/4k/terra-relevo.jpg';
import terraRugosidadeHD from '../assets/planets/4k/terra-rugosidade.jpg';
import terraHD from '../assets/planets/4k/terra.jpg';
import terraNuvensHD from '../assets/planets/4k/terra-nuvens.jpg';
import mercurioHD from '../assets/planets/4k/mercurio.jpg';
import venusHD from '../assets/planets/4k/venus.jpg';
import marteHD from '../assets/planets/4k/marte.jpg';
import jupiterHD from '../assets/planets/4k/jupiter.jpg';
import saturnoHD from '../assets/planets/4k/saturno.jpg';
import uranoHD from '../assets/planets/4k/urano.jpg';
import netunoHD from '../assets/planets/4k/netuno.jpg';

export const PLANET_TEXTURES = {
  terra,
  terraNuvens,
  mercurio,
  venus,
  marte,
  jupiter,
  saturno,
  urano,
  netuno,
  saturnoAneis,
  mercurioRelevo,
  marteRelevo,
  terraRelevo,
  terraRugosidade,
};

// versão 4096 × 2048 (HD): mais nitidez ao dar zoom num ponto da superfície
export const PLANET_TEXTURES_HD = {
  terra: terraHD,
  terraNuvens: terraNuvensHD,
  mercurio: mercurioHD,
  venus: venusHD,
  marte: marteHD,
  jupiter: jupiterHD,
  saturno: saturnoHD,
  urano: uranoHD,
  netuno: netunoHD,
  saturnoAneis,
  mercurioRelevo: mercurioRelevoHD,
  marteRelevo: marteRelevoHD,
  terraRelevo: terraRelevoHD,
  terraRugosidade: terraRugosidadeHD,
};
