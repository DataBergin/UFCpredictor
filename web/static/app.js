// ============================================
// Sports Predict — Apple-style Frontend
// ============================================

const socket = io();

// State
let currentSport = 'ufc';
let isRunning = false;

// DOM refs
const terminal      = document.getElementById('terminal-content');
const statusPill    = document.getElementById('status-pill');
const statusText    = document.getElementById('status-text');
const statusDot     = document.getElementById('status-dot');
const connDot       = document.getElementById('connection-dot');
const btnCancel     = document.getElementById('btn-cancel');
const btnClear      = document.getElementById('btn-clear');
const footerSport   = document.getElementById('footer-sport');

// Sport-specific labels
const SPORT_CONFIG = {
    ufc: {
        labelA: 'Fighter A', labelB: 'Fighter B',
        placeholderA: 'Islam Makhachev', placeholderB: 'Charles Oliveira',
        defaultData: 'data/raw/ufcstats_fights.csv',
        color: '#ff3b30',
    },
    soccer: {
        labelA: 'Home Team', labelB: 'Away Team',
        placeholderA: 'France', placeholderB: 'Argentina',
        defaultData: 'data/raw/world_cup_matches.csv',
        color: '#34c759',
    },
    mlb: {
        labelA: 'Home Team', labelB: 'Away Team',
        placeholderA: 'NYY', placeholderB: 'BOS',
        defaultData: 'data/raw/mlb_games.csv',
        color: '#007aff',
    },
};

// ============================================
// Segmented Control (animated pill indicator)
// ============================================

const segments  = document.querySelectorAll('.segment');
const indicator = document.querySelector('.segment-indicator');

function updateIndicator() {
    const active = document.querySelector('.segment.active');
    if (!active || !indicator) return;
    indicator.style.width = active.offsetWidth + 'px';
    indicator.style.transform = `translateX(${active.offsetLeft - active.parentElement.offsetLeft - 2}px)`;
}

// Set initial indicator position after render
requestAnimationFrame(() => {
    requestAnimationFrame(updateIndicator);
});
window.addEventListener('resize', updateIndicator);

segments.forEach(seg => {
    seg.addEventListener('click', () => {
        if (isRunning) return;

        segments.forEach(s => s.classList.remove('active'));
        seg.classList.add('active');
        currentSport = seg.dataset.sport;

        updateIndicator();

        const cfg = SPORT_CONFIG[currentSport];
        document.getElementById('label-a').textContent = cfg.labelA;
        document.getElementById('label-b').textContent = cfg.labelB;
        document.getElementById('entity-a').placeholder = cfg.placeholderA;
        document.getElementById('entity-b').placeholder = cfg.placeholderB;
        document.getElementById('data-path').placeholder = cfg.defaultData;

        // UFC-specific options
        const ufcOpts = document.getElementById('ufc-options');
        if (currentSport === 'ufc') {
            ufcOpts.classList.remove('hidden');
        } else {
            ufcOpts.classList.add('hidden');
        }

        // Update footer badge
        footerSport.textContent = currentSport.toUpperCase();
    });
});

// ============================================
// Socket.IO — stream output
// ============================================

socket.on('output', (data) => {
    const text = data.text || '';

    const span = document.createElement('span');
    span.className = 'line';

    if (text.startsWith('$')) {
        span.classList.add('cmd');
    } else if (text.includes('[ERROR]') || text.includes('[CANCELLED]')) {
        span.classList.add('error');
    } else if (text.includes('[COMPLETED]') || text.includes('Pipeline complete') || text.includes('Done!')) {
        span.classList.add('success');
    }

    span.textContent = text;
    terminal.appendChild(span);

    // Smooth auto-scroll
    const scrollEl = terminal.parentElement;
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: 'smooth' });

    if (data.done) {
        setRunning(false);
    }
});

socket.on('connect', () => {
    connDot.classList.remove('disconnected');
    connDot.title = 'Connected';
    if (!isRunning) statusText.textContent = 'Ready';
});

socket.on('disconnect', () => {
    connDot.classList.add('disconnected');
    connDot.title = 'Disconnected';
    statusText.textContent = 'Disconnected';
});

// ============================================
// Command dispatch
// ============================================

function runCommand(action, args = {}) {
    if (isRunning) return;
    setRunning(true);
    clearTerminal();

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

    if (running) {
        statusPill.classList.add('running');
        statusText.textContent = 'Running';
    } else {
        statusPill.classList.remove('running');
        statusText.textContent = 'Ready';
    }

    // Disable/enable buttons
    document.querySelectorAll('.btn-primary, .btn-secondary').forEach(btn => {
        btn.disabled = running;
    });
    segments.forEach(s => {
        s.style.pointerEvents = running ? 'none' : 'auto';
        s.style.opacity = running ? '0.5' : '1';
    });
}

function clearTerminal() {
    terminal.innerHTML = '';
}

// ============================================
// Button handlers
// ============================================

// Predict
document.getElementById('btn-predict').addEventListener('click', () => {
    const entityA = document.getElementById('entity-a').value;
    const entityB = document.getElementById('entity-b').value;

    if (!entityA || !entityB) {
        clearTerminal();
        const msg = document.createElement('span');
        msg.className = 'line muted';
        msg.textContent = 'Enter both names to run a prediction.';
        terminal.appendChild(msg);
        return;
    }

    runCommand('predict', {
        entity_a: entityA,
        entity_b: entityB,
        date: document.getElementById('predict-date').value || '',
        model: document.getElementById('predict-model').value || '',
    });
});

// Scrape
document.getElementById('btn-scrape').addEventListener('click', () => {
    runCommand('scrape');
});

// Train
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
        clearTerminal();
        const msg = document.createElement('span');
        msg.className = 'line muted';
        msg.textContent = 'Enter a directory path or RSS URL to ingest.';
        terminal.appendChild(msg);
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
    const msg = document.createElement('span');
    msg.className = 'line muted';
    msg.textContent = 'Select a sport and run a command to get started.';
    terminal.appendChild(msg);
});

// Enter key on entity inputs triggers predict
document.querySelectorAll('#entity-a, #entity-b').forEach(input => {
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('btn-predict').click();
    });
});

// ============================================
// Load models on startup
// ============================================

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
