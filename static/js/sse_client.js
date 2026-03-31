/**
 * SSE Client -- Server-Sent Events + polling fallback.
 * Connects to /api/sessions/{id}/logs for live events.
 * Falls back to polling /logs/history if SSE drops.
 * Historical logs loaded from DB on init.
 */

let _sseConnection = null;
let _sseSessionId = null;
let _historyLoaded = false;
let _lastLogStage = null;
let _logPollTimer = null;
let _lastLogCount = 0;  // Track how many logs we have to detect new ones

/**
 * Load historical logs from DB, then connect SSE for live events.
 */
async function initLogs(sessionId) {
    _sseSessionId = sessionId;
    _historyLoaded = false;
    _lastLogCount = 0;

    // Load historical logs from REST endpoint
    try {
        const resp = await fetch(`/api/sessions/${sessionId}/logs/history`);
        const json = await resp.json();
        if (json.success && json.data && json.data.length > 0) {
            const logBody = document.getElementById('log-body');
            if (logBody) logBody.innerHTML = ''; // Clear hardcoded initial message
            let lastStage = null;
            for (const log of json.data) {
                // Insert stage separator on stage change in historical logs
                if (log.stage && log.stage !== lastStage) {
                    lastStage = log.stage;
                    _lastLogStage = log.stage;
                    _appendStageSeparator(log.stage);
                }
                appendLog(log.level || 'info', log.message || '');
            }
            _lastLogCount = json.data.length;
            _historyLoaded = true;
        }
    } catch (e) {
        console.warn('Failed to load historical logs:', e);
    }

    // Connect SSE for live events
    connectSSE(sessionId);

    // Start log polling fallback (catches logs if SSE drops)
    startLogPolling(sessionId);
}

function connectSSE(sessionId) {
    if (_sseConnection) {
        _sseConnection.close();
    }

    _sseSessionId = sessionId;
    const url = `/api/sessions/${sessionId}/logs`;

    _sseConnection = new EventSource(url);

    _sseConnection.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleSSEEvent(data);
        } catch (e) {
            console.warn('SSE parse error:', e);
        }
    };

    _sseConnection.onerror = (err) => {
        const state = _sseConnection ? _sseConnection.readyState : -1;
        // readyState: 0=CONNECTING, 1=OPEN, 2=CLOSED
        console.warn('[SSE] Connection error, readyState=' + state + '. Polling fallback active.');
        if (state === 2) {
            console.warn('[SSE] Connection CLOSED by server/proxy. Relying on log polling fallback.');
        }
    };

    _sseConnection.onopen = () => {
        console.log('[SSE] Connection opened for session ' + sessionId.substring(0, 8));
        if (!_historyLoaded) {
            appendLog('info', 'Live log stream connected');
        }
    };
}

function disconnectSSE() {
    if (_sseConnection) {
        _sseConnection.close();
        _sseConnection = null;
    }
    stopLogPolling();
}

function handleSSEEvent(data) {
    if (data.type === 'log') {
        // Insert stage separator when stage changes
        if (data.stage && data.stage !== _lastLogStage) {
            _lastLogStage = data.stage;
            _appendStageSeparator(data.stage);
        }
        appendLog(data.level || 'info', data.message || '');
        _lastLogCount++;
    } else if (data.type === 'progress') {
        updateProgress(data.stage, data.pct, data.message);
    }
}

/**
 * Polling fallback -- fetches logs from DB every 5s.
 * Only appends NEW logs (compares count with _lastLogCount).
 * This ensures logs appear even if SSE connection is broken.
 */
function startLogPolling(sessionId) {
    stopLogPolling();
    _logPollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/sessions/${sessionId}/logs/history`);
            const json = await resp.json();
            if (!json.success || !json.data) return;

            const allLogs = json.data;
            if (allLogs.length > _lastLogCount) {
                // New logs arrived that SSE missed -- polling fallback caught them
                const newLogs = allLogs.slice(_lastLogCount);
                console.log('[SSE-FALLBACK] Polling caught ' + newLogs.length + ' new logs (SSE missed them)');
                for (const log of newLogs) {
                    if (log.stage && log.stage !== _lastLogStage) {
                        _lastLogStage = log.stage;
                        _appendStageSeparator(log.stage);
                    }
                    appendLog(log.level || 'info', log.message || '');
                }
                _lastLogCount = allLogs.length;
            }
        } catch (e) {
            // Silently ignore polling errors
        }
    }, 5000);
}

function stopLogPolling() {
    if (_logPollTimer) {
        clearInterval(_logPollTimer);
        _logPollTimer = null;
    }
}

const _stageLabelMap = {
    'stage1_upload': 'Stage 1: PDF Upload',
    'stage2_mapping': 'Stage 2: LLM Mapper',
    'stage3_cleaning': 'Stage 3: Cleaner',
    'stage4_metadata': 'Stage 4: Metadata',
    'stage5_watermark': 'Stage 5: Watermark',
    'stage6_images': 'Stage 6: Image CDN',
    'stage7_batch': 'Stage 7: LaTeX + Tagging',
    'stage8_duplicates': 'Stage 8: Duplicate Check',
    'stage9_quality': 'Stage 9: Quality Verifier',
};

function _appendStageSeparator(stageName) {
    const logBody = document.getElementById('log-body');
    if (!logBody) return;
    const label = _stageLabelMap[stageName] || stageName;
    const sep = document.createElement('div');
    sep.className = 'log-entry separator';
    sep.innerHTML = `<span class="log-separator">--- ${escapeHtml(label)} ---</span>`;
    logBody.appendChild(sep);
    logBody.scrollTop = logBody.scrollHeight;
}

/**
 * Append a log entry to the log pane.
 */
function appendLog(level, message) {
    const logBody = document.getElementById('log-body');
    if (!logBody) return;

    const levelMap = {
        'info': 'INFO',
        'success': 'OK',
        'warning': 'WARN',
        'error': 'ERR',
    };

    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    entry.innerHTML = `
        <span class="log-level">[${levelMap[level] || level.toUpperCase()}]</span>
        <span class="log-msg">${escapeHtml(message)}</span>
    `;

    logBody.appendChild(entry);

    // Auto-scroll to bottom
    logBody.scrollTop = logBody.scrollHeight;

    // Limit to 500 entries
    while (logBody.children.length > 500) {
        logBody.removeChild(logBody.firstChild);
    }
}

/**
 * Update a progress bar by stage name.
 */
function updateProgress(stage, pct, message) {
    const bar = document.querySelector(`[data-progress="${stage}"] .progress-fill`);
    if (bar) {
        bar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    }
    const msg = document.querySelector(`[data-progress="${stage}"] .progress-msg`);
    if (msg && message) {
        msg.textContent = message;
    }
}

function clearLogs() {
    const logBody = document.getElementById('log-body');
    if (logBody) {
        logBody.innerHTML = '';
    }
    _lastLogCount = 0;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
