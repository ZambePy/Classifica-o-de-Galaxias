// Dados dos exemplos com modelo 3D ilustrativo (galáxias, nebulosas e os 8 planetas do Sistema Solar).
// view: 'face' (de frente) | 'tilt' (inclinado) | 'edge' (de lado) | 'front' (nebulosas "planas", vistas de frente)
// flat: cenas que só fazem sentido de frente (giro limitado)
// Distâncias: pesquisa/01-galaxias.md, pesquisa/02-nebulosas.md e fontes conferidas em 24/09/2026 (ver AUDITORIA.md).
export const OBJECTS = {
  // ---------------- galáxias
  m87: {
    kind: 'elliptical', p: { R: 7, axis: [1, 0.95, 0.92], jet: true, globs: 600 }, view: 'tilt', seed: 87,
    pt: { name: 'M87', dist: 'Cerca de 55 milhões de anos-luz', fact: 'Gigante no centro do Aglomerado de Virgem. Foi a primeira galáxia a ter a foto do seu buraco negro, em 2019. O risco azul é um jato de matéria saindo do centro.' },
    en: { name: 'M87', dist: 'About 55 million light-years', fact: 'A giant at the centre of the Virgo Cluster. It was the first galaxy to have its black hole photographed, in 2019. The blue streak is a jet of matter shooting out of the centre.' },
  },
  m49: {
    kind: 'elliptical', p: { R: 6.5, axis: [1, 0.8, 0.85] }, view: 'tilt', seed: 49,
    pt: { name: 'M49', dist: 'Cerca de 56 milhões de anos-luz', fact: 'Foi a primeira galáxia do Aglomerado de Virgem a ser descoberta. É uma bola enorme de estrelas antigas.' },
    en: { name: 'M49', dist: 'About 56 million light-years', fact: 'The first galaxy of the Virgo Cluster ever discovered. A huge ball of old stars.' },
  },
  fuso: {
    kind: 'lenticular', view: 'edge', seed: 5866,
    pt: { name: 'Galáxia do Fuso (NGC 5866)', dist: 'Cerca de 45 milhões de anos-luz', fact: 'Um disco visto quase de lado, sem braços, com uma faixa fina de poeira bem no meio.' },
    en: { name: 'Spindle Galaxy (NGC 5866)', dist: 'About 45 million light-years', fact: 'A disc seen almost side-on, with no arms and a thin stripe of dust right in the middle.' },
  },
  viaLactea: {
    kind: 'milkyway', view: 'tilt', seed: 1,
    pt: { name: 'Via Láctea', dist: 'Estamos dentro dela', fact: 'A nossa galáxia, com cerca de 100 mil anos-luz de uma ponta à outra. O ponto amarelo mostra mais ou menos onde fica o Sol.' },
    en: { name: 'Milky Way', dist: 'We are inside it', fact: 'Our own galaxy, about 100,000 light-years across. The yellow dot shows roughly where the Sun is.' },
  },
  ngc1300: {
    kind: 'barred', p: { R: 10, bar: 4, turns: 0.5 }, view: 'face', seed: 1300,
    pt: { name: 'NGC 1300', dist: 'Cerca de 70 milhões de anos-luz', fact: 'Tem uma barra de estrelas enorme no centro, fácil de ver. Os braços saem das pontas da barra.' },
    en: { name: 'NGC 1300', dist: 'About 70 million light-years', fact: 'It has a huge, easy-to-see bar of stars in the middle. The arms start at the ends of the bar.' },
  },
  andromeda: {
    kind: 'spiral', p: { R: 11, turns: 1.3, spread: 0.45, bulge: 2.2 }, view: 'edge', tiltOverride: 0.35, seed: 31,
    pt: { name: 'Andrômeda', dist: 'Cerca de 2,5 milhões de anos-luz', fact: 'A grande galáxia mais próxima da nossa. Em céu bem escuro, dá para ver a olho nu como uma mancha de luz.' },
    en: { name: 'Andromeda', dist: 'About 2.5 million light-years', fact: 'The nearest large galaxy to ours. Under a very dark sky you can see it with the naked eye as a faint smudge.' },
  },
  bode: {
    kind: 'spiral', p: { R: 9, turns: 1.2, spread: 0.35, bulge: 2 }, view: 'tilt', seed: 81,
    pt: { name: 'Galáxia de Bode (M81)', dist: 'Cerca de 12 milhões de anos-luz', fact: 'Uma espiral bem organizada e brilhante. É vizinha da Galáxia do Charuto.' },
    en: { name: "Bode's Galaxy (M81)", dist: 'About 12 million light-years', fact: 'A bright, well-organised spiral. It is a neighbour of the Cigar Galaxy.' },
  },
  cataVento: {
    kind: 'spiral', p: { R: 12, arms: 3, turns: 0.65, spread: 0.5, bulge: 1, knots: 220 }, view: 'face', seed: 101,
    pt: { name: 'Galáxia do Cata-vento (M101)', dist: 'Cerca de 25 milhões de anos-luz', fact: 'Enorme: mede cerca de 170 mil anos-luz, bem maior que a Via Láctea. Os braços são soltos e cheios de regiões rosadas onde nascem estrelas.' },
    en: { name: 'Pinwheel Galaxy (M101)', dist: 'About 25 million light-years', fact: 'Huge: about 170,000 light-years across, much bigger than the Milky Way. Its loose arms are full of pink star-forming regions.' },
  },
  triangulo: {
    kind: 'spiral', p: { R: 8, turns: 0.7, spread: 0.6, bulge: 0.7, knots: 160 }, view: 'tilt', seed: 33,
    pt: { name: 'Galáxia do Triângulo (M33)', dist: 'Cerca de 3 milhões de anos-luz', fact: 'A terceira maior galáxia do nosso grupo de galáxias vizinhas, com metade do tamanho da Via Láctea.' },
    en: { name: 'Triangulum Galaxy (M33)', dist: 'About 3 million light-years', fact: 'The third-largest galaxy in our local group, about half the size of the Milky Way.' },
  },
  redemoinho: {
    kind: 'whirlpool', view: 'face', seed: 51,
    pt: { name: 'Galáxia do Redemoinho (M51)', dist: 'Cerca de 31 milhões de anos-luz', fact: 'Está puxando uma galáxia menor, que aparece presa na ponta de um dos braços.' },
    en: { name: 'Whirlpool Galaxy (M51)', dist: 'About 31 million light-years', fact: 'It is pulling on a smaller galaxy that looks stuck to the end of one of its arms.' },
  },
  sombrero: {
    kind: 'sombrero', view: 'edge', tiltOverride: 0.12, seed: 104,
    pt: { name: 'Galáxia do Sombrero (M104)', dist: 'Cerca de 30 milhões de anos-luz', fact: 'Tem um centro grande e brilhante e um anel escuro de poeira em volta, como a aba de um chapéu.' },
    en: { name: 'Sombrero Galaxy (M104)', dist: 'About 30 million light-years', fact: 'It has a big, bright centre and a dark ring of dust around it, like the brim of a hat.' },
  },
  agulha: {
    kind: 'needle', view: 'edge', seed: 4565,
    pt: { name: 'Galáxia da Agulha (NGC 4565)', dist: 'Entre 30 e 50 milhões de anos-luz', fact: 'Vista exatamente de lado, parece uma agulha fina no céu, com uma linha de poeira no meio.' },
    en: { name: 'Needle Galaxy (NGC 4565)', dist: '30 to 50 million light-years', fact: 'Seen exactly side-on, it looks like a thin needle in the sky, with a line of dust through the middle.' },
  },
  antenas: {
    kind: 'antennae', view: 'tilt', seed: 4038,
    pt: { name: 'Galáxias Antenas', dist: 'Cerca de 45 milhões de anos-luz', fact: 'Duas galáxias batendo uma na outra. As longas caudas de estrelas lembram as antenas de um inseto.' },
    en: { name: 'Antennae Galaxies', dist: 'About 45 million light-years', fact: 'Two galaxies crashing into each other. Their long tails of stars look like insect antennae.' },
  },
  ratos: {
    kind: 'mice', view: 'tilt', seed: 4676,
    pt: { name: 'Galáxias dos Ratos', dist: 'Cerca de 300 milhões de anos-luz', fact: 'Duas galáxias se juntando, cada uma com uma cauda comprida, como dois ratinhos.' },
    en: { name: 'Mice Galaxies', dist: 'About 300 million light-years', fact: 'Two galaxies merging, each with a long tail, like two little mice.' },
  },
  lmc: {
    kind: 'lmc', view: 'tilt', seed: 160,
    pt: { name: 'Grande Nuvem de Magalhães', dist: 'Cerca de 160 mil anos-luz', fact: 'Uma galáxia pequena que gira em volta da Via Láctea. Aparece no céu do hemisfério sul. A mancha rosa forte é a Nebulosa da Tarântula.' },
    en: { name: 'Large Magellanic Cloud', dist: 'About 160,000 light-years', fact: 'A small galaxy that orbits the Milky Way, visible in the southern sky. The bright pink patch is the Tarantula Nebula.' },
  },
  charuto: {
    kind: 'cigar', view: 'front', seed: 82,
    pt: { name: 'Galáxia do Charuto (M82)', dist: 'Cerca de 12 milhões de anos-luz', fact: 'Forma estrelas muito rápido. Os jatos vermelhos são gás sendo empurrado para fora pelo centro agitado.' },
    en: { name: 'Cigar Galaxy (M82)', dist: 'About 12 million light-years', fact: 'It forms stars very fast. The red plumes are gas being pushed out by its busy centre.' },
  },

  // ---------------- nebulosas
  orion: {
    kind: 'emission', p: { youngStars: 40, trapezium: true }, view: 'front', flat: true, seed: 42,
    pt: { name: 'Nebulosa de Órion', dist: 'Cerca de 1.300 anos-luz', fact: 'Um berçário de estrelas que dá para ver a olho nu, na "espada" da constelação de Órion.' },
    en: { name: 'Orion Nebula', dist: 'About 1,300 light-years', fact: 'A stellar nursery you can see with the naked eye, in the "sword" of the Orion constellation.' },
  },
  aguia: {
    kind: 'eagle', view: 'front', flat: true, seed: 16,
    pt: { name: 'Nebulosa da Águia', dist: 'Cerca de 6.500 anos-luz', fact: 'Onde ficam os famosos Pilares da Criação: colunas de gás e poeira onde estrelas estão nascendo.' },
    en: { name: 'Eagle Nebula', dist: 'About 6,500 light-years', fact: 'Home of the famous Pillars of Creation: columns of gas and dust where stars are being born.' },
  },
  pilares: {
    kind: 'pillars', view: 'front', flat: true, seed: 1995,
    pt: { name: 'Pilares da Criação', dist: 'Cerca de 6.500 anos-luz', fact: 'Três colunas enormes de gás e poeira dentro da Nebulosa da Águia, onde estrelas estão nascendo. Ficaram famosas numa foto do Hubble de 1995 e foram fotografadas de novo pelo James Webb em 2022. As bordas brilham porque recebem a luz de estrelas jovens próximas.' },
    en: { name: 'Pillars of Creation', dist: 'About 6,500 light-years', fact: 'Three huge columns of gas and dust inside the Eagle Nebula, where stars are being born. They became famous in a 1995 Hubble photo and were photographed again by James Webb in 2022. Their edges glow because nearby young stars light them up.' },
  },
  lagoa: {
    kind: 'emission', p: { colors: ['#ff4d6d', '#ff6f91', '#ffb3c7', '#e0435f', '#8fe3ff'], blobs: 9, R: 9, darkLane: true }, view: 'front', flat: true, seed: 8,
    pt: { name: 'Nebulosa da Lagoa', dist: 'Entre 4 mil e 6 mil anos-luz', fact: 'Tão brilhante que dá para ver a olho nu em céu escuro. A faixa escura no meio lembra um lago.' },
    en: { name: 'Lagoon Nebula', dist: '4,000 to 6,000 light-years', fact: 'So bright you can see it with the naked eye under a dark sky. The dark lane in the middle looks like a lagoon.' },
  },
  pleiades: {
    kind: 'pleiades', view: 'front', flat: true, seed: 45,
    pt: { name: 'Plêiades', dist: 'Cerca de 440 anos-luz', fact: 'Um grupo de estrelas jovens e azuis passando por uma nuvem de poeira, que reflete a luz delas.' },
    en: { name: 'Pleiades', dist: 'About 440 light-years', fact: 'A group of young blue stars passing through a dust cloud that reflects their light.' },
  },
  bruxa: {
    kind: 'witchhead', view: 'front', flat: true, seed: 2118,
    pt: { name: 'Nebulosa Cabeça da Bruxa', dist: 'Cerca de 900 a 1.000 anos-luz', fact: 'Poeira iluminada pela estrela Rigel (o ponto brilhante), com formato parecido com o perfil de uma bruxa.' },
    en: { name: 'Witch Head Nebula', dist: 'About 900 to 1,000 light-years', fact: 'Dust lit by the star Rigel (the bright dot), shaped like the profile of a witch.' },
  },
  cavalo: {
    kind: 'horsehead', view: 'front', flat: true, seed: 33,
    pt: { name: 'Nebulosa Cabeça de Cavalo', dist: 'Cerca de 1.375 anos-luz', fact: 'Uma nuvem escura em forma de cabeça de cavalo, na frente de um fundo vermelho que brilha.' },
    en: { name: 'Horsehead Nebula', dist: 'About 1,375 light-years', fact: 'A dark cloud shaped like a horse head, in front of a glowing red background.' },
  },
  carvao: {
    kind: 'coalsack', view: 'front', flat: true, seed: 590,
    pt: { name: 'Saco de Carvão', dist: 'Cerca de 590 anos-luz', fact: 'Uma mancha escura na Via Láctea, ao lado do Cruzeiro do Sul (as 4 estrelas brilhantes), a constelação da bandeira do Brasil.' },
    en: { name: 'Coalsack', dist: 'About 590 light-years', fact: 'A dark patch in the Milky Way next to the Southern Cross (the 4 bright stars), the constellation on the Brazilian flag.' },
  },
  anel: {
    kind: 'ring', view: 'face', seed: 57,
    pt: { name: 'Nebulosa do Anel (M57)', dist: 'Cerca de 2.600 anos-luz', fact: 'Um anel de gás soltado por uma estrela que está chegando ao fim da vida. A estrela fica bem no centro.' },
    en: { name: 'Ring Nebula (M57)', dist: 'About 2,600 light-years', fact: 'A ring of gas released by a star reaching the end of its life. The star sits right in the middle.' },
  },
  helice: {
    kind: 'helix', view: 'face', seed: 7293,
    pt: { name: 'Nebulosa da Hélice', dist: 'Cerca de 655 anos-luz', fact: 'Uma das nebulosas planetárias mais próximas da Terra. Vista de frente, parece um olho gigante.' },
    en: { name: 'Helix Nebula', dist: 'About 655 light-years', fact: 'One of the planetary nebulae closest to Earth. Seen face-on, it looks like a giant eye.' },
  },
  olhoGato: {
    kind: 'catseye', view: 'tilt', seed: 6543,
    pt: { name: 'Olho de Gato', dist: 'Cerca de 3.300 anos-luz', fact: 'Tem camadas de gás umas dentro das outras, soltas pela estrela em momentos diferentes.' },
    en: { name: "Cat's Eye", dist: 'About 3,300 light-years', fact: 'It has layers of gas nested inside each other, released by the star at different times.' },
  },
  caranguejo: {
    kind: 'crab', view: 'tilt', seed: 1054,
    pt: { name: 'Nebulosa do Caranguejo', dist: 'Cerca de 6.500 anos-luz', fact: 'Resto de uma estrela que explodiu no ano de 1054. No centro gira uma estrela minúscula e muito rápida, que pisca cerca de 30 vezes por segundo.' },
    en: { name: 'Crab Nebula', dist: 'About 6,500 light-years', fact: 'The remains of a star that exploded in the year 1054. At its centre spins a tiny, very fast star that flashes about 30 times per second.' },
  },
  veu: {
    kind: 'veil', view: 'tilt', seed: 2400,
    pt: { name: 'Nebulosa do Véu', dist: 'Cerca de 2.400 anos-luz', fact: 'Fios de gás de uma explosão que aconteceu há 10 a 20 mil anos e ainda está se espalhando.' },
    en: { name: 'Veil Nebula', dist: 'About 2,400 light-years', fact: 'Threads of gas from an explosion 10,000 to 20,000 years ago that is still spreading out.' },
  },
  bolha: {
    kind: 'bubble', view: 'tilt', seed: 7635,
    pt: { name: 'Nebulosa da Bolha', dist: 'Entre 7 mil e 11 mil anos-luz', fact: 'Uma bolha soprada pelo vento de uma estrela gigante, cerca de 44 vezes mais pesada que o Sol.' },
    en: { name: 'Bubble Nebula', dist: '7,000 to 11,000 light-years', fact: 'A bubble blown by the wind of a giant star, about 44 times heavier than the Sun.' },
  },
  crescente: {
    kind: 'crescent', view: 'tilt', seed: 6888,
    pt: { name: 'Nebulosa Crescente', dist: 'Cerca de 5 mil anos-luz', fact: 'Uma casca de gás empurrada pelo vento forte de uma estrela muito quente, que fica no centro.' },
    en: { name: 'Crescent Nebula', dist: 'About 5,000 light-years', fact: 'A shell of gas pushed out by the strong wind of a very hot star at its centre.' },
  },
  // ---------------- planetas do Sistema Solar (kind: 'planet', ver three/planets.js)
  // Números: NASA Planetary Fact Sheet (conferido em 24/09/2026). tiltDeg = inclinação do eixo;
  // acima de 90° o planeta gira "ao contrário" (Vênus e Urano), então não precisa de outra marcação.
  mercurio: {
    kind: 'planet', group: 'rocky', camDist: 17,
    planet: { tex: 'mercurio', tiltDeg: 0.03, dayHours: 4222.6, rough: 1, normal: 'mercurioRelevo', normalScale: 1.3 },
    pt: { name: 'Mercúrio', dist: 'Cerca de 58 milhões de km do Sol', size: '4.879 km (pouco mais de um terço da Terra)', short: 'O menor e o mais perto do Sol', fact: 'O menor planeta e o mais perto do Sol. É cheio de crateras e um ano lá dura só 88 dias.' },
    en: { name: 'Mercury', dist: 'About 58 million km from the Sun', size: '4,879 km (just over a third of Earth)', short: 'The smallest and closest to the Sun', fact: 'The smallest planet and the closest to the Sun. It is covered in craters and a year there lasts only 88 days.' },
  },
  venus: {
    kind: 'planet', group: 'rocky', camDist: 17,
    planet: { tex: 'venus', tiltDeg: 177.4, dayHours: 2802, atmo: '#ffd9a0', atmoStrength: 0.9 },
    pt: { name: 'Vênus', dist: 'Cerca de 108 milhões de km do Sol', size: '12.104 km (quase igual à Terra)', short: 'O mais quente de todos', fact: 'O planeta mais quente: cerca de 460 °C, porque as nuvens grossas prendem o calor. Ele gira ao contrário da maioria dos planetas.' },
    en: { name: 'Venus', dist: 'About 108 million km from the Sun', size: '12,104 km (almost the same as Earth)', short: 'The hottest of all', fact: 'The hottest planet: about 460 °C, because its thick clouds trap the heat. It spins the opposite way to most planets.' },
  },
  terra: {
    kind: 'planet', group: 'rocky', camDist: 17,
    planet: { tex: 'terra', tiltDeg: 23.4, dayHours: 24, clouds: 'terraNuvens', atmo: '#5fa8ff', atmoStrength: 1.25, normal: 'terraRelevo', normalScale: 1.2, roughMap: 'terraRugosidade' },
    pt: { name: 'Terra', dist: 'Cerca de 150 milhões de km do Sol', size: '12.756 km', short: 'Nossa casa', fact: 'Nossa casa e o único lugar onde sabemos que existe vida. Cerca de 70% da superfície é coberta de água.' },
    en: { name: 'Earth', dist: 'About 150 million km from the Sun', size: '12,756 km', short: 'Our home', fact: 'Our home and the only place where we know life exists. About 70% of its surface is covered by water.' },
  },
  marte: {
    kind: 'planet', group: 'rocky', camDist: 17,
    planet: { tex: 'marte', tiltDeg: 25.2, dayHours: 24.7, atmo: '#ff9a6a', atmoStrength: 0.5, atmoSize: 1.04, normal: 'marteRelevo', normalScale: 1.3 },
    pt: { name: 'Marte', dist: 'Cerca de 228 milhões de km do Sol', size: '6.792 km (cerca de metade da Terra)', short: 'O planeta vermelho', fact: 'É vermelho por causa da ferrugem no chão. Um dia lá dura quase o mesmo que aqui: pouco mais de 24 horas e meia.' },
    en: { name: 'Mars', dist: 'About 228 million km from the Sun', size: '6,792 km (about half of Earth)', short: 'The red planet', fact: 'It is red because of rust in its soil. A day there is almost the same as here: just over 24 and a half hours.' },
  },
  jupiter: {
    kind: 'planet', group: 'gas', camDist: 17,
    planet: { tex: 'jupiter', tiltDeg: 3.1, dayHours: 9.9, atmo: '#ffe3b8', atmoStrength: 0.4 },
    pt: { name: 'Júpiter', dist: 'Cerca de 778 milhões de km do Sol', size: '142.984 km (11 Terras lado a lado)', short: 'O maior de todos', fact: 'O maior planeta: caberiam mais de 1.000 Terras dentro dele. A mancha vermelha é uma tempestade maior que a Terra.' },
    en: { name: 'Jupiter', dist: 'About 778 million km from the Sun', size: '142,984 km (11 Earths side by side)', short: 'The biggest of all', fact: 'The biggest planet: more than 1,000 Earths would fit inside it. The red spot is a storm bigger than Earth.' },
  },
  saturno: {
    kind: 'planet', group: 'gas', camDist: 24, tiltOverride: 0.32,
    planet: { tex: 'saturno', tiltDeg: 26.7, dayHours: 10.7, rings: 'saturn', atmo: '#ffe9c4', atmoStrength: 0.35 },
    pt: { name: 'Saturno', dist: 'Cerca de 1,4 bilhão de km do Sol', size: '120.536 km sem os anéis (mais de 9 Terras)', short: 'O dos anéis famosos', fact: 'Os anéis são feitos de pedaços de gelo e rocha. Ele é tão leve para o seu tamanho que flutuaria na água.' },
    en: { name: 'Saturn', dist: 'About 1.4 billion km from the Sun', size: '120,536 km without the rings (over 9 Earths)', short: 'The one with famous rings', fact: 'Its rings are made of chunks of ice and rock. It is so light for its size that it would float on water.' },
  },
  urano: {
    kind: 'planet', group: 'ice', camDist: 23, tiltOverride: 0.2,
    planet: { tex: 'urano', tiltDeg: 97.8, yawDeg: 50, dayHours: 17.2, rings: 'uranus', atmo: '#9ff3ff', atmoStrength: 0.8 },
    pt: { name: 'Urano', dist: 'Cerca de 2,9 bilhões de km do Sol', size: '51.118 km (4 Terras)', short: 'Gira deitado', fact: 'Gira quase deitado, inclinado cerca de 98°. A cor azul-esverdeada vem de um gás chamado metano.' },
    en: { name: 'Uranus', dist: 'About 2.9 billion km from the Sun', size: '51,118 km (4 Earths)', short: 'Spins on its side', fact: 'It spins almost on its side, tilted about 98°. Its blue-green colour comes from a gas called methane.' },
  },
  netuno: {
    kind: 'planet', group: 'ice', camDist: 17,
    planet: { tex: 'netuno', tiltDeg: 28.3, dayHours: 16.1, atmo: '#5d8cff', atmoStrength: 0.9 },
    pt: { name: 'Netuno', dist: 'Cerca de 4,5 bilhões de km do Sol', size: '49.528 km (quase 4 Terras)', short: 'O mais distante', fact: 'O planeta mais longe do Sol. Tem os ventos mais fortes do Sistema Solar, com mais de 2.000 km/h.' },
    en: { name: 'Neptune', dist: 'About 4.5 billion km from the Sun', size: '49,528 km (almost 4 Earths)', short: 'The farthest away', fact: 'The planet farthest from the Sun. It has the strongest winds in the Solar System, over 2,000 km/h.' },
  },
};

// ordem do Sol para fora
export const PLANET_IDS = ['mercurio', 'venus', 'terra', 'marte', 'jupiter', 'saturno', 'urano', 'netuno'];

// ângulo inicial (rotação em X) para cada tipo de vista
export const VIEW_TILT = { face: 1.35, tilt: 0.75, edge: 0.06, front: 0 };
