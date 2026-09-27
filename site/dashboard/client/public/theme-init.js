// Aplica o tema antes do React carregar, evitando o "piscar" de cores.
(function () {
  var pref = 'system';
  try {
    pref = localStorage.getItem('pg-tema') || 'system';
  } catch (e) {}
  var dark = pref === 'dark' || (pref !== 'light' && (!window.matchMedia || window.matchMedia('(prefers-color-scheme: dark)').matches));
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
})();
