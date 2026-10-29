/* ============================================================================
   MedSphere — API configuration
   ----------------------------------------------------------------------------
   Single source of truth for the backend URL. Previously each page carried its
   own: two dead ngrok tunnels, a "your-production-api-url.com" placeholder and
   a hardcoded 127.0.0.1.

   To point the deployed site at your API, set PRODUCTION_API_BASE below to the
   Render web service URL (no trailing slash), e.g.
       https://medsphere-api.onrender.com
   ========================================================================= */

(function (global) {
  'use strict';

  // >>> EDIT THIS LINE ONLY. <<<
  var PRODUCTION_API_BASE = 'https://medicine-analyzer-qd2t.onrender.com';

  // >>> DO NOT EDIT THE LINE BELOW. <<<
  // This is a fixed sentinel, not configuration. It records the original
  // placeholder value so the code can tell "the URL was never set" apart from
  // "the URL is set but the server is down", and report the right thing. If it
  // is edited to match PRODUCTION_API_BASE, every failure is misreported as an
  // unset placeholder.
  var DEFAULT_PRODUCTION_API_BASE = 'https://medsphere-api.onrender.com';

  var LOCAL_API_BASE = 'http://127.0.0.1:8000';

  var LOCAL_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0', ''];
  var isLocal = LOCAL_HOSTS.indexOf(global.location.hostname) !== -1 ||
                global.location.protocol === 'file:';

  var base = isLocal ? LOCAL_API_BASE : PRODUCTION_API_BASE;

  // Escape hatches for pointing the site at a different API without editing and
  // re-pushing. Handy while working out whether a problem is the frontend or
  // the backend.
  //
  //   local dev  : ?api=https://host      (query param, convenient)
  //   deployed   : localStorage.setItem('medsphere_api_base', 'https://host')
  //
  // The query param is honoured ONLY when running locally. Accepting it on the
  // deployed site would mean a link like ...?api=https://attacker.example could
  // make a genuine-looking MedSphere page render attacker-supplied drug data.
  // localStorage needs devtools access, so it cannot be triggered by a link.
  try {
    var override = null;
    if (isLocal) {
      var match = /[?&]api=([^&]+)/.exec(global.location.search);
      if (match) override = decodeURIComponent(match[1]);
    }
    if (!override && global.localStorage) {
      override = global.localStorage.getItem('medsphere_api_base');
    }
    if (override && /^https?:\/\/.+/i.test(override)) {
      base = override.replace(/\/+$/, '');
      if (global.console) console.info('[MedSphere] API override in use:', base);
    }
  } catch (storageBlocked) {
    /* localStorage can throw when cookies are blocked; the default is fine. */
  }

  var usingDefaultProdUrl = !isLocal && base === DEFAULT_PRODUCTION_API_BASE;

  /**
   * Build a full API URL.
   * @param {string} path   e.g. '/api/drug_details'
   * @param {Object} [params] query parameters; undefined/null values are skipped
   */
  function apiUrl(path, params) {
    var url = base.replace(/\/+$/, '') + '/' + String(path).replace(/^\/+/, '');
    if (!params) return url;

    var parts = [];
    Object.keys(params).forEach(function (key) {
      var value = params[key];
      if (value === undefined || value === null || value === '') return;
      parts.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
    });
    return parts.length ? url + '?' + parts.join('&') : url;
  }

  /* ------------------------------------------------------------------------
     Cold starts. The free Render instance sleeps after 15 minutes idle and
     takes up to a minute or two to wake. While it wakes, requests fail at the
     network layer or come back as 502/503 from Render's edge. Instead of
     showing an error, retry for up to WAKE_LIMIT_MS and tell the user why.
     ------------------------------------------------------------------------ */
  var WAKE_LIMIT_MS = 150000;
  var banner = null;
  var bannerTimer = null;

  function showWakeBanner(startedAt) {
    if (!global.document || !document.body) return;
    if (!banner) {
      banner = document.createElement('div');
      banner.className = 'ms-wake';
      banner.setAttribute('role', 'status');
      banner.setAttribute('aria-live', 'polite');
      document.body.appendChild(banner);
    }
    function render() {
      var seconds = Math.round((Date.now() - startedAt) / 1000);
      banner.textContent = 'Waking the server (free hosting sleeps when idle). ' +
        'This can take up to a minute... ' + seconds + 's';
    }
    render();
    banner.hidden = false;
    if (!bannerTimer) bannerTimer = global.setInterval(render, 1000);
  }

  function hideWakeBanner() {
    if (bannerTimer) { global.clearInterval(bannerTimer); bannerTimer = null; }
    if (banner) banner.hidden = true;
  }

  function sleep(ms) {
    return new Promise(function (resolve) { global.setTimeout(resolve, ms); });
  }

  /* A reply that did not come from our API: no JSON body, status from the
     edge while the instance boots. */
  function isWaking(response, payload) {
    return !payload && (response.status === 502 || response.status === 503 ||
                        response.status === 504);
  }

  function apiError(message, status, payload) {
    var error = new Error(message);
    error.status = status || 0;
    error.payload = payload || null;
    return error;
  }

  /**
   * Fetch JSON from the API, surfacing the backend's own error message.
   *
   * The API replies with {"error": "...", ...} and a 4xx/5xx status for
   * failures, so read the body before deciding what to throw. The thrown
   * Error carries .status and .payload, so pages can show the "did you mean"
   * suggestions that come with a 404.
   */
  async function apiFetch(path, params, options) {
    var url = apiUrl(path, params);
    var startedAt = Date.now();
    var attempt = 0;

    try {
      while (true) {
        var response = null;
        var payload = null;
        try {
          response = await fetch(url, options);
          try { payload = await response.json(); } catch (parseError) { payload = null; }
        } catch (networkError) {
          response = null;
        }

        var waking = response === null || isWaking(response, payload);
        if (waking && !usingDefaultProdUrl && !isLocal &&
            Date.now() - startedAt < WAKE_LIMIT_MS) {
          attempt += 1;
          if (attempt >= 2) showWakeBanner(startedAt);
          await sleep(Math.min(2000 * attempt, 8000));
          continue;
        }

        if (response === null) {
          if (usingDefaultProdUrl) {
            throw apiError(
              'Could not reach the API at ' + base + '. This is still the ' +
              'placeholder URL. Set PRODUCTION_API_BASE in Frontend/config.js to ' +
              'your own API address and redeploy.'
            );
          }
          throw apiError(isLocal
            ? 'Could not reach the local API at ' + base + '. Is the backend running?'
            : 'Could not reach the server. Please try again in a minute.');
        }

        if (!response.ok) {
          // A wrong API URL does not always fail at the network layer.
          // Unclaimed *.onrender.com hostnames answer with a plain 404.
          if (usingDefaultProdUrl) {
            throw apiError(
              'The API URL is still the placeholder (' + base + '), which returned ' +
              response.status + '. Set PRODUCTION_API_BASE in Frontend/config.js to ' +
              'your own Render URL and redeploy.', response.status
            );
          }
          // Our API always sends {"error": "..."} with a failure status.
          if (payload && payload.error) {
            throw apiError(payload.error, response.status, payload);
          }
          if (response.status === 404) {
            throw apiError('No server is responding at ' + base + '. The API may not be ' +
                           'deployed yet, or its URL may have changed.', 404);
          }
          throw apiError('Request failed with status ' + response.status, response.status);
        }
        if (payload && payload.error) {
          throw apiError(payload.error, response.status, payload);
        }
        return payload;
      }
    } finally {
      hideWakeBanner();
    }
  }

  /* Start waking the server as soon as any page opens, so it is often awake
     by the time the user has typed a drug name. */
  if (!isLocal && !usingDefaultProdUrl && global.fetch) {
    try {
      fetch(apiUrl('/healthz'), { mode: 'cors', cache: 'no-store' }).catch(function () {});
    } catch (ignored) { /* best effort */ }
  }

  /**
   * Render "did you mean" buttons for a 404 that came with suggestions.
   * onPick(name) is called with the drug name the user chose.
   */
  function renderSuggestions(container, error, onPick) {
    var list = error && error.payload && error.payload.suggestions;
    if (!container || !list || !list.length) return false;
    var wrap = document.createElement('div');
    wrap.className = 'ms-didyoumean';
    var label = document.createElement('span');
    label.className = 'ms-didyoumean__label';
    label.textContent = 'Did you mean:';
    wrap.appendChild(label);
    list.forEach(function (item) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'ms-chip ms-didyoumean__item';
      button.textContent = item.label.toLowerCase() === item.name.toLowerCase()
        ? item.name
        : item.label + ' \u2192 ' + item.name;
      button.addEventListener('click', function () { onPick(item.name); });
      wrap.appendChild(button);
    });
    container.appendChild(wrap);
    return true;
  }

  /**
   * Attach an autocomplete list to a text input, fed by /api/suggest.
   * onPick(name) runs when a suggestion is chosen (click or Enter).
   */
  function attachAutocomplete(input, onPick) {
    if (!input || input.dataset.msAutocomplete) return;
    input.dataset.msAutocomplete = '1';
    input.setAttribute('autocomplete', 'off');
    input.setAttribute('aria-autocomplete', 'list');

    var host = input.parentNode;
    if (global.getComputedStyle(host).position === 'static') host.style.position = 'relative';
    var list = document.createElement('ul');
    list.className = 'ms-ac';
    list.setAttribute('role', 'listbox');
    list.hidden = true;
    host.appendChild(list);

    var timer = null;
    var items = [];
    var active = -1;
    var latest = 0;

    function close() { list.hidden = true; active = -1; }

    function highlight(index) {
      active = index;
      Array.prototype.forEach.call(list.children, function (li, i) {
        li.classList.toggle('is-active', i === index);
        li.setAttribute('aria-selected', String(i === index));
      });
    }

    function choose(item) {
      input.value = item.name;
      close();
      if (onPick) onPick(item.name);
    }

    function render(results) {
      items = results;
      list.innerHTML = '';
      results.forEach(function (item) {
        var li = document.createElement('li');
        li.setAttribute('role', 'option');
        var main = document.createElement('span');
        main.className = 'ms-ac__name';
        main.textContent = item.label;
        li.appendChild(main);
        var meta = document.createElement('span');
        meta.className = 'ms-ac__meta';
        meta.textContent = item.label.toLowerCase() === item.name.toLowerCase()
          ? (item.approved ? 'approved' : 'not approved')
          : item.kind + ' of ' + item.name;
        li.appendChild(meta);
        li.addEventListener('mousedown', function (event) {
          event.preventDefault();   // keep focus; blur would close first
          choose(item);
        });
        list.appendChild(li);
      });
      list.hidden = results.length === 0;
      active = -1;
    }

    input.addEventListener('input', function () {
      var query = input.value.trim();
      global.clearTimeout(timer);
      if (query.length < 2) { close(); return; }
      timer = global.setTimeout(async function () {
        var ticket = ++latest;
        try {
          var data = await apiFetch('/api/suggest', { q: query });
          if (ticket === latest && document.activeElement === input) {
            render((data && data.suggestions) || []);
          }
        } catch (ignored) { close(); }
      }, 150);
    });

    input.addEventListener('keydown', function (event) {
      if (list.hidden || !items.length) return;
      if (event.key === 'ArrowDown') {
        event.preventDefault(); highlight((active + 1) % items.length);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault(); highlight((active - 1 + items.length) % items.length);
      } else if (event.key === 'Enter' && active >= 0) {
        event.preventDefault(); event.stopImmediatePropagation(); choose(items[active]);
      } else if (event.key === 'Escape') {
        close();
      }
    });
    input.addEventListener('blur', function () { global.setTimeout(close, 100); });
  }

  /** Escape text for safe use inside innerHTML templates. */
  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  var DISCLAIMER = 'For information only. Similar drugs are not automatically safe ' +
    'substitutes. Ask a doctor or pharmacist before changing any medicine.';

  global.MedSphereConfig = {
    apiBase: base,
    isLocal: isLocal,
    apiUrl: apiUrl,
    apiFetch: apiFetch,
    renderSuggestions: renderSuggestions,
    attachAutocomplete: attachAutocomplete,
    escapeHtml: escapeHtml,
    disclaimer: DISCLAIMER,
    usingDefaultProdUrl: usingDefaultProdUrl
  };
})(window);
