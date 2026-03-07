/**
 * SSE Client — Generic Server-Sent Events connection manager.
 * Connects to /api/sessions/{id}/logs and dispatches events to the log pane.
 * Historical logs are loaded from DB first, then SSE streams live events.
 */

let _sseConnection = null;
let _sseSessionId = null;
let _historyLoaded = false;
let _lastLogStage = null;

/**
 * Load historical logs from DB, then connect SSE for live events.
 */
async function initLogs(sessionId) {
    _sseSessionId = sessionId;
    _historyLoaded = false;

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
            _historyLoaded = true;
        }
    } catch (e) {
        console.warn('Failed to load historical logs:', e);
    }

    // Then connect SSE for live events
    connectSSE(sessionId);
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
        console.warn('SSE connection error, will retry...');
    };

    _sseConnection.onopen = () => {
        if (!_historyLoaded) {
            appendLog('info', 'SSE connected — live log stream active');
        }
    };
}

function disconnectSSE() {
    if (_sseConnection) {
        _sseConnection.close();
        _sseConnection = null;
    }
}

function handleSSEEvent(data) {
    if (data.type === 'log') {
        // Insert stage separator when stage changes
        if (data.stage && data.stage !== _lastLogStage) {
            _lastLogStage = data.stage;
            _appendStageSeparator(data.stage);
        }
        appendLog(data.level || 'info', data.message || '');
    } else if (data.type === 'progress') {
        updateProgress(data.stage, data.pct, data.message);
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
    sep.innerHTML = `<span class="log-separator">━━━ ${escapeHtml(label)} ━━━</span>`;
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
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
