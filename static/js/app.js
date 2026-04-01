/**
 * App.js — Global application logic, navigation, helpers.
 */

// ── API helpers ──

async function apiCall(method, url, body = null) {
    const opts = {
        method,
        headers: {'Content-Type': 'application/json'},
    };
    if (body) {
        opts.body = JSON.stringify(body);
    }
    const resp = await fetch(url, opts);
    // Guard against HTML error pages (502, OOM restart, proxy errors)
    const ct = resp.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
        const text = (await resp.text()).slice(0, 120);
        return { success: false, error: { code: 'NON_JSON', message: `Server returned ${resp.status} (${text})` } };
    }
    return resp.json();
}

async function apiPost(url, body = null) {
    return apiCall('POST', url, body);
}

async function apiGet(url) {
    return apiCall('GET', url);
}

// ── Stage actions ──

async function advanceStage(sessionId) {
    const result = await apiPost(`/api/sessions/${sessionId}/advance`);
    if (result.success) {
        // Redirect to session page WITHOUT stage query param so it picks up
        // the new current_stage from DB. Using reload() would preserve any
        // ?stage= param and show the same stage again.
        window.location.href = `/session/${sessionId}`;
    } else {
        appendLog('error', result.error?.message || 'Failed to advance stage');
    }
}

// ── Sidebar helpers ──

/**
 * Update the sidebar quarantine count by adding delta to current value.
 */
function updateSidebarQuarantineCount(delta) {
    const el = document.getElementById('sidebar-quarantine-count');
    if (el) {
        const current = parseInt(el.textContent) || 0;
        el.textContent = current + delta;
    }
}

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {
    // Ctrl+L — focus log pane
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        const logBody = document.getElementById('log-body');
        if (logBody) logBody.scrollTop = logBody.scrollHeight;
    }
});
