// ============================================
// Sports Predict — Frontend Logic
// ============================================

const socket = io();

// State
let currentSport = 'ufc';
let isRunning = false;

// DOM refs
const terminal    = document.getElementById('terminal-content');
const statusText  = document.getElementById('status-text');
const statusSport = document.getElementById('status-sport');
const statusBar   = document.getElementById('status-bar');
const btnCancel   = document.getElementById('btn-cancel');
const btnClear    = document.getElementById('btn-clear');

// Labels that change per sport
const SPORT_CONFIG = {
    ufc: {
        labelA: 'Fighter A', labelB: 'Fighter B',
        placeholderA: 'Islam Makhachev', placeholderB: 'Charles Oliveira',
        defaultData: 'data/raw/ufcstats_fights.csv',
    },
    soccer: {
        labelA: 'Home Team', labelB: 'Away Team',
        placeholderA: 'France', placeholderB: 'Argentina',
        defaultData: 'data/raw/world_cup_matches.csv',
    },
    mlb: {
        labelA: 'Home Team', labelB: 'Away Team',
        placeholderA: 'NYY', placeholderB: 'BOS',
        defaultData: 'data/raw/mlb_games.csv',
    },
};

// ------------------------------------------------------------------
// Sport tabs
// ------------------------------------------------------------------
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (isRunning) return;

        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentSport = tab.dataset.sport;

        const cfg = SPORT_CONFIG[currentSport];
        document.getElementById('label-a').textContent = cfg.labelA;
        document.getElementById('label-b').textContent = cfg.labelB;
        document.getElementById('entity-a').placeholder = cfg.placeholderA;
        document.getElementById('entity-b').placeholder = cfg.placeholderB;
        document.getElementById('data-path').placeholder = cfg.defaultData;

        // Show/hide UFC-specific options
        const ufcOpts = document.getElementById('ufc-options');
        if (currentSport === 'ufc') {
            ufcOpts.classList.remove('hidden');
        } else {
            ufcOpts.classList.add('hidden');
        }

        statusSport.textContent = currentSport.toUpperCase();
    });
});

// ------------------------------------------------------------------
// Socket output handling
// ------------------------------------------------------------------
socket.on('output', (data) => {
    const text = data.text || '';
    const span = document.createElement('span');

    // Color code certain lines
    if (text.startsWith('$')) {
        span.className = 'cmd';
    } else if (text.includes('[ERROR]') || text.includes('[CANCELLED]')) {
        span.className = 'error';
    } else if (text.includes('[COMPLETED]') || text.includes('Pipeline complete') || text.includes('Done!')) {
        span.className = 'success';
    }

    span.textContent = text;
    terminal.appendChild(span);

    // Auto-scroll
    terminal.parentElement.scrollTop = terminal.parentElement.scrollHeight;

    if (data.done) {
        setRunning(false);
    }
});

socket.on('disconnect', () => {
    statusText.textContent = 'Disconnected';
});

socket.on('connect', () => {
    if (!isRunning) statusText.textContent = 'Connected';
});

// ------------------------------------------------------------------
// Command execution
// ------------------------------------------------------------------
function runCommand(action, args = {}) {
    if (isRunning) return;
    setRunning(true);
    clearTerminal();

    // For UFC, use the richer UFC-specific CLI when predicting
    let actualAction = action;
    if (currentSport === 'ufc' && action === 'predict') {
        actualAction = 'ufc_predict';
        args.rounds = document.getElementById('predict-rounds').value;
    }
    if (currentSport === 'ufc' && action === 'run_full') {
        actualAction = 'ufc_run_full';
    }
    if (currentSport === 'ufc' && action === 'scrape') {
        actualAction = 'ufc_scrape';
    }

    socket.emit('run_command', {
        sport: currentSport,
        action: actualAction,
        args: args,
    });
}

function setRunning(running) {
    isRunning = running;
    btnCancel.disabled = !running;
    statusText.textContent = running ? 'Running...' : 'Idle';
    statusBar.className = running ? 'status-bar running' : 'status-bar';

    // Disable/enable action buttons
    document.querySelectorAll('.btn:not(.btn-sm)').forEach(btn => {
        if (btn.id !== 'btn-cancel') btn.disabled = running;
    });
    document.querySelectorAll('.tab').forEach(t => {
        t.style.pointerEvents = running ? 'none' : 'auto';
        t.style.opacity = running ? '0.5' : '1';
    });
}

function clearTerminal() {
    terminal.innerHTML = '';
}

// ------------------------------------------------------------------
// Button handlers
// ------------------------------------------------------------------

// Predict
document.getElementById('btn-predict').addEventListener('click', () => {
    const entityA = document.getElementById('entity-a').value;
    const entityB = document.getElementById('entity-b').value;

    if (!entityA || !entityB) {
        terminal.innerHTML = '<span class="error">Enter both names to predict.</span>';
        return;
    }

    const args = {
        entity_a: entityA,
        entity_b: entityB,
        date: document.getElementById('predict-date').value || '',
        model: document.getElementById('predict-model').value || '',
    };

    runCommand('predict', args);
});

// Scrape
document.getElementById('btn-scrape').addEventListener('click', () => {
    runCommand('scrape');
});

// Train (run-full)
document.getElementById('btn-train').addEventListener('click', () => {
    const dataPath = document.getElementById('data-path').value ||
                     SPORT_CONFIG[currentSport].defaultData;
    runCommand('run_full', { data: dataPath });
});

// Ingest
document.getElementById('btn-ingest').addEventListener('click', () => {
    const sourceDir = document.getElementById('ingest-dir').value;
    const rssUrl = document.getElementById('ingest-rss').value;

    if (!sourceDir && !rssUrl) {
        terminal.innerHTML = '<span class="error">Enter a directory or RSS URL to ingest.</span>';
        return;
    }

    runCommand('ingest', { source_dir: sourceDir, rss_url: rssUrl });
});

// Cancel
btnCancel.addEventListener('click', () => {
    socket.emit('cancel_command');
});

// Clear
btnClear.addEventListener('click', () => {
    clearTerminal();
    terminal.innerHTML = '<span class="muted">Ready.</span>';
});

// Enter key triggers predict
document.querySelectorAll('#entity-a, #entity-b').forEach(input => {
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('btn-predict').click();
    });
});

// ------------------------------------------------------------------
// Load models list on startup
// ------------------------------------------------------------------
fetch('/api/models')
    .then(r => r.json())
    .then(models => {
        const select = document.getElementById('predict-model');
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = `${m.name} (${m.size_mb} MB)`;
            select.appendChild(opt);
        });
    })
    .catch(() => {});
