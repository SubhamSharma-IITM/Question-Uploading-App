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

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {
    // Ctrl+L — focus log pane
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        const logBody = document.getElementById('log-body');
        if (logBody) logBody.scrollTop = logBody.scrollHeight;
    }
});
