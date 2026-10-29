/* ============================================================================
   MedSphere — shared page chrome
   ----------------------------------------------------------------------------
   Injects the homepage's background stack and sidebar into every page, so all
   pages share one definition instead of three hand-copied variants that drift
   apart (and three that had no sidebar at all).

   Both injections are idempotent: if a page already contains the markup — as
   index.html, compare_page.html and drugdetail1.html do — it is left alone and
   only the behaviour is wired up. That keeps the hand-built homepage untouched.

   Usage: <script src="site-chrome.js" defer></script>
          <body data-ms-page="compare">      <!-- marks the active nav item -->
   ========================================================================= */

(function () {
  'use strict';

  /* The navigation, defined once. `key` matches body[data-ms-page]. */
  var NAV_ITEMS = [
    { key: 'home',    href: 'index.html',        icon: 'bx-home-smile',  label: 'Home' },
    { key: 'details', href: 'drugdetail1.html',  icon: 'bx-capsule',     label: 'Drug Details' },
    { key: 'alt',     href: 'firstpage.html',    icon: 'bx-analyse',     label: 'Alternatives' },
    { key: 'compare', href: 'compare_page.html', icon: 'bx-git-compare', label: 'Comparison' },
    { key: 'about',   href: 'about.html',        icon: 'bxs-contact',    label: 'About' }
  ];

  /* Faint line drawings of two real molecules, ibuprofen and paracetamol, in
     the skeletal style the structure viewer uses. They replace the stock
     "rising squares" animation that was copied from a CSS template. */
  var MOLECULES =
    '<svg class="ms-mol ms-mol--a" viewBox="-110 -160 230 330" aria-hidden="true" focusable="false">' +
    '<path d="M0.0 -40.0L34.6 -20.0M34.6 -20.0L34.6 20.0M34.6 20.0L0.0 40.0M0.0 40.0L-34.6 20.0M-34.6 20.0L-34.6 -20.0M-34.6 -20.0L0.0 -40.0M3.2 -29.3L23.8 -17.5M23.8 17.5L3.2 29.3M-27.0 11.9L-27.0 -11.9M0.0 -40.0L0.0 -80.0M0.0 -80.0L-34.6 -100.0M-34.6 -100.0L-34.6 -140.0M-34.6 -100.0L-69.3 -80.0M0.0 40.0L0.0 80.0M0.0 80.0L34.6 100.0M0.0 80.0L-34.6 100.0M34.6 100.0L34.6 134.0M28.6 100.0L28.6 134.0M34.6 100.0L63.3 83.5"/>' +
    '<text x="27.6" y="158">O</text><text x="67.3" y="86">OH</text></svg>' +
    '<svg class="ms-mol ms-mol--b" viewBox="-60 -110 160 270" aria-hidden="true" focusable="false">' +
    '<path d="M0.0 -40.0L34.6 -20.0M34.6 -20.0L34.6 20.0M34.6 20.0L0.0 40.0M0.0 40.0L-34.6 20.0M-34.6 20.0L-34.6 -20.0M-34.6 -20.0L0.0 -40.0M3.2 -29.3L23.8 -17.5M23.8 17.5L3.2 29.3M-27.0 11.9L-27.0 -11.9M0.0 -40.0L0.0 -70.0M0.0 40.0L0.0 68.0M8.0 84.0L34.6 100.0M34.6 100.0L34.6 134.0M28.6 100.0L28.6 134.0M34.6 100.0L69.3 80.0"/>' +
    '<text x="-16" y="-76">HO</text><text x="-8" y="84">N</text><text x="-8" y="104">H</text>' +
    '<text x="27.6" y="152">O</text></svg>';

  function addMolecules(background) {
    if (!background || background.querySelector('.ms-molecules')) return;
    var layer = document.createElement('div');
    layer.className = 'ms-molecules';
    layer.setAttribute('aria-hidden', 'true');
    layer.innerHTML = MOLECULES;
    background.appendChild(layer);
  }

  function buildBackground() {
    var background = document.createElement('div');
    /* ms-bg-fixed takes the layer out of document flow. bg.css sizes
       .background at 100vh, which is what the homepage hero wants but would
       otherwise push a content page's first screen entirely below the fold. */
    background.className = 'background ms-bg-fixed';
    background.setAttribute('aria-hidden', 'true');

    var area = document.createElement('div');
    area.className = 'area';

    /* The homepage's soft white glow, so content pages get the same depth. */
    var glow = document.createElement('div');
    glow.className = 'layer-blur';

    background.appendChild(area);
    background.appendChild(glow);
    return background;
  }

  function buildSidebar(activeKey) {
    var nav = document.createElement('nav');

    var sidebar = document.createElement('div');
    sidebar.className = 'sidebar';

    /* --- logo / toggle --- */
    var logoContent = document.createElement('div');
    logoContent.className = 'logo-content';

    var logo = document.createElement('div');
    logo.className = 'logo';

    var img = document.createElement('img');
    img.src = 'logo.png';
    img.alt = '';
    img.className = 'log';

    var wordmark = document.createElement('span');
    wordmark.textContent = 'MedSphere';

    var toggle = document.createElement('i');
    toggle.className = 'bx bx-menu';
    toggle.id = 'btn';
    toggle.setAttribute('role', 'button');
    toggle.setAttribute('tabindex', '0');
    toggle.setAttribute('aria-label', 'Toggle navigation');
    toggle.setAttribute('aria-expanded', 'false');

    logo.appendChild(img);
    logo.appendChild(wordmark);
    logo.appendChild(toggle);
    logoContent.appendChild(logo);

    /* --- links --- */
    var list = document.createElement('ul');
    NAV_ITEMS.forEach(function (item) {
      var li = document.createElement('li');
      if (item.key === activeKey) li.className = 'active';

      var link = document.createElement('a');
      link.href = item.href;
      if (item.key === activeKey) link.setAttribute('aria-current', 'page');

      var icon = document.createElement('i');
      icon.className = 'bx ' + item.icon;
      icon.setAttribute('aria-hidden', 'true');

      var label = document.createElement('span');
      label.className = 'nav-item';
      label.textContent = item.label;

      /* Keeps the label readable while the rail is collapsed. */
      link.setAttribute('aria-label', item.label);
      link.appendChild(icon);
      link.appendChild(label);
      li.appendChild(link);
      list.appendChild(li);
    });

    sidebar.appendChild(logoContent);
    sidebar.appendChild(list);
    nav.appendChild(sidebar);
    return nav;
  }

  /* Wire the hamburger for whichever sidebar is on the page, injected or not. */
  function wireSidebarToggle() {
    var button = document.querySelector('#btn');
    var sidebar = document.querySelector('.sidebar');
    if (!button || !sidebar) return;

    function toggle() {
      var isOpen = sidebar.classList.toggle('active');
      document.querySelectorAll('.nav-item').forEach(function (item) {
        item.classList.toggle('active', isOpen);
      });
      button.setAttribute('aria-expanded', String(isOpen));
      document.body.classList.toggle('ms-sidebar-open', isOpen);
    }

    button.addEventListener('click', toggle);
    button.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggle();
      }
    });

    /* Collapse on Escape so keyboard users are not trapped in an open rail. */
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && sidebar.classList.contains('active')) toggle();
    });
  }

  /* Marks the active item on pages that already ship their own sidebar. */
  function markActive(activeKey) {
    var target = NAV_ITEMS.filter(function (i) { return i.key === activeKey; })[0];
    if (!target) return;

    document.querySelectorAll('.sidebar ul li').forEach(function (li) {
      var link = li.querySelector('a');
      if (!link) return;
      var href = (link.getAttribute('href') || '').split('/').pop();
      if (href === target.href) {
        li.classList.add('active');
        link.setAttribute('aria-current', 'page');
      }
    });
  }

  function hidePreloader() {
    var preloader = document.getElementById('preloader');
    var content = document.getElementById('main-content');
    if (!preloader) return;

    preloader.classList.add('hidden');
    preloader.style.opacity = '0';
    preloader.style.transition = 'opacity 0.5s ease';
    window.setTimeout(function () {
      preloader.style.display = 'none';
      if (content) content.style.display = 'block';
    }, 500);
  }

  function init() {
    var activeKey = document.body.getAttribute('data-ms-page') || '';

    /* Only inject what is missing. The background may legitimately live inside
       a wrapper (index.html nests it in #main-content), so search the whole
       document rather than assuming it is a child of <body>. */
    if (!document.querySelector('.background')) {
      document.body.insertBefore(buildBackground(), document.body.firstChild);
    }
    addMolecules(document.querySelector('.background'));

    /* The sidebar is position:fixed, so where it sits in the DOM has no visual
       effect; putting it first keeps it early in the tab order. */
    if (!document.querySelector('.sidebar')) {
      document.body.insertBefore(buildSidebar(activeKey), document.body.firstChild);
    } else {
      markActive(activeKey);
    }

    wireSidebarToggle();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* The preloader must clear even if an image or font never resolves. */
  window.addEventListener('load', hidePreloader);
  window.setTimeout(hidePreloader, 4000);
})();
