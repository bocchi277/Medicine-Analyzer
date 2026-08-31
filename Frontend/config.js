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

  // >>> EDIT THIS after the API is deployed. <<<
  // Still the default value below? Then the deployed site is pointing at a URL
  // that probably isn't yours, and every request will fail.
  var PRODUCTION_API_BASE = 'https://medsphere-api.onrender.com';
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

  /**
   * Fetch JSON from the API, surfacing the backend's own error message.
   *
   * The API replies with {"error": "..."} and a 4xx/5xx status for failures,
   * so read the body before deciding what to throw.
   */
  async function apiFetch(path, params, options) {
    var url = apiUrl(path, params);
    var response;

    try {
      response = await fetch(url, options);
    } catch (networkError) {
      // Two very different causes produce the same browser error, so name both.
      if (usingDefaultProdUrl) {
        throw new Error(
          'Could not reach the API at ' + base + '. This is still the ' +
          'placeholder URL — set PRODUCTION_API_BASE in Frontend/config.js to ' +
          'your own API address and redeploy.'
        );
      }
      throw new Error(
        'Could not reach the server. It may be starting up — free instances ' +
        'sleep after 15 minutes of inactivity and take about a minute to wake. ' +
        'Try again shortly.'
      );
    }

    var payload = null;
    try {
      payload = await response.json();
    } catch (parseError) {
      payload = null;
    }

    if (!response.ok) {
      // A wrong API URL does not always fail at the network layer. Unclaimed
      // *.onrender.com hostnames answer with a 404, which would otherwise
      // surface as a bare "Request failed with status 404" and send you looking
      // for a bug in the backend that isn't there.
      if (usingDefaultProdUrl) {
        throw new Error(
          'The API URL is still the placeholder (' + base + '), which returned ' +
          response.status + '. Set PRODUCTION_API_BASE in Frontend/config.js to ' +
          'your own Render URL and redeploy.'
        );
      }
      throw new Error(
        (payload && payload.error) || ('Request failed with status ' + response.status)
      );
    }
    if (payload && payload.error) {
      throw new Error(payload.error);
    }
    return payload;
  }

  global.MedSphereConfig = {
    apiBase: base,
    isLocal: isLocal,
    apiUrl: apiUrl,
    apiFetch: apiFetch,
    usingDefaultProdUrl: usingDefaultProdUrl
  };
})(window);
