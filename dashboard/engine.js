// Particles JS Config
particlesJS("particles-js", {
    "particles": { "number": { "value": 50 }, "color": { "value": "#66FCF1" }, "opacity": { "value": 0.2 }, "size": { "value": 2 }, "line_linked": { "enable": true, "distance": 150, "color": "#66FCF1", "opacity": 0.1, "width": 1 }, "move": { "enable": true, "speed": 1 } }
});

// Globals
const HTTP_PORT = window.location.port ? parseInt(window.location.port) : 8080;
const WS_PORT = HTTP_PORT + 1000;
let ws;
let eventCount = 0;
let globalNodesData = [];

// Chart.js Instance
let kdeChart;
let selectedSymbol = null;

// Vis Network Instance
let network;
let nodesDataset = new vis.DataSet();
let edgesDataset = new vis.DataSet();

function initChart() {
    const ctx = document.getElementById('kdeChart').getContext('2d');
    kdeChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Density', data: [], borderColor: '#66FCF1', backgroundColor: 'rgba(102, 252, 241, 0.2)', fill: true, tension: 0.4, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: true, ticks: { color: '#C5C6C7', maxTicksLimit: 5 } }, y: { display: false } }, animation: { duration: 0 } }
    });
}

function initNetwork() {
    const container = document.getElementById('network-container');
    const data = { nodes: nodesDataset, edges: edgesDataset };
    const options = {
        nodes: { shape: 'dot', font: { color: '#FFF', face: 'Orbitron', size: 12 }, borderWidth: 2, shadow: true },
        edges: { color: { color: 'rgba(102,252,241,0.2)', highlight: '#FF2A2A' }, smooth: { type: 'continuous' } },
        physics: { barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 150 } },
        interaction: { hover: true }
    };
    network = new vis.Network(container, data, options);
    
    network.on('click', function (params) {
        if (params.nodes.length > 0) {
            selectNode(params.nodes[0]);
        }
    });
}

function selectNode(sym) {
    selectedSymbol = sym;
    document.getElementById('kde-subtitle').innerText = `NODE: ${sym}`;
    updateChartForSelection();
    
    // Highlight list item
    document.querySelectorAll('.node-item').forEach(el => el.classList.remove('active'));
    const item = document.getElementById(`list-${sym}`);
    if(item) item.classList.add('active');
}

function updateChartForSelection() {
    if(!selectedSymbol) return;
    const node = globalNodesData.find(n => n.symbol === selectedSymbol);
    if(node && node.kde_x && node.kde_x.length > 0) {
        kdeChart.data.labels = node.kde_x.map(x => x.toFixed(5));
        kdeChart.data.datasets[0].data = node.kde_y;
        kdeChart.update();
    }
}

function connectWebSocket() {
    ws = new WebSocket(`ws://localhost:${WS_PORT}`);
    const statusEl = document.getElementById('ws-status');

    ws.onopen = () => { statusEl.innerText = "TELEMETRY LINK SECURE"; statusEl.style.color = "var(--cyan)"; };

    ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        if (data.type === 'heartbeat') {
            document.getElementById('volume-progress').style.width = `${data.volume_progress}%`;
            document.getElementById('volume-text').innerText = `Ingesting: ${data.volume_progress.toFixed(1)}%`;
        } else if (data.type === 'physics_update') {
            globalNodesData = data.nodes;
            updateUI(data);
        }
    };

    ws.onclose = () => {
        statusEl.innerText = "LINK SEVERED. RECONNECTING..."; statusEl.style.color = "var(--red)";
        setTimeout(connectWebSocket, 2000);
    };
}

function updateUI(data) {
    // Top Nodes List
    const listEl = document.getElementById('top-nodes-list');
    listEl.innerHTML = '';
    let sortedNodes = [...data.nodes].sort((a, b) => Math.abs(b.sigma) - Math.abs(a.sigma));
    
    sortedNodes.slice(0, 15).forEach(node => {
        const color = Math.abs(node.sigma) > 2.0 ? 'var(--red)' : 'var(--text-bright)';
        const dir = node.force > 0 ? '↑' : '↓';
        const activeClass = selectedSymbol === node.symbol ? 'active' : '';
        const html = `<div class="node-item ${activeClass}" id="list-${node.symbol}" onclick="selectNode('${node.symbol}')">
            <span class="node-sym">${node.symbol}</span>
            <span class="node-sig" style="color:${color}">${dir} ${node.sigma.toFixed(2)}σ</span>
        </div>`;
        listEl.insertAdjacentHTML('beforeend', html);
    });

    if(!selectedSymbol && sortedNodes.length > 0) {
        selectNode(sortedNodes[0].symbol);
    } else {
        updateChartForSelection();
    }

    // Vis Network Updates
    const visNodes = [];
    data.nodes.forEach(n => {
        let nColor = '#45A29E'; // default
        let nSize = 10;
        if(Math.abs(n.sigma) > 3.0) { nColor = '#FF2A2A'; nSize = 25; }
        else if(Math.abs(n.sigma) > 2.0) { nColor = '#FF9900'; nSize = 18; }
        
        visNodes.push({
            id: n.symbol,
            label: n.symbol,
            color: { background: '#0a0a0f', border: nColor },
            size: nSize
        });
    });
    nodesDataset.update(visNodes);

    if (data.edges) {
        const visEdges = data.edges.map(e => ({
            id: `${e.source}-${e.target}`,
            from: e.source,
            to: e.target,
            value: e.weight
        }));
        edgesDataset.update(visEdges);
    }

    // Events
    if (data.events && data.events.length > 0) {
        const feed = document.getElementById('event-feed');
        const emptyState = document.getElementById('empty-feed');
        if(emptyState) emptyState.style.display = 'none';
        
        data.events.forEach(ev => {
            eventCount++;
            document.getElementById('event-count').innerText = eventCount;
            const html = `<div class="event-card">
                <span class="event-sym">⚠️ ${ev.symbol} [${ev.direction}]</span>
                <span>σ: ${ev.sigma.toFixed(2)} | Ripple: ${ev.ripple_target}</span><br>
                <span style="color:var(--cyan)">Stop: ${(ev.stop_dist*100).toFixed(2)}% | Risk: ${ev.risk.toFixed(1)}%</span><br>
                <span style="color: #FF9900; font-size: 0.75rem;">VACUUM: ${ev.vacuum.toFixed(5)} | COST: ${ev.friction.toFixed(5)}</span>
            </div>`;
            feed.insertAdjacentHTML('afterbegin', html);
        });
    }

    // Shadow Ledger Update
    if (data.shadow_balance !== undefined) {
        document.getElementById('shadow-balance').innerText = `$${data.shadow_balance.toFixed(2)}`;
    }

    if (data.active_shadows !== undefined) {
        const activeContainer = document.getElementById('shadow-active');
        if (data.active_shadows.length === 0) {
            activeContainer.innerHTML = '<p class="empty-text" id="empty-active-shadow">NO ACTIVE TRADES</p>';
        } else {
            activeContainer.innerHTML = '';
            data.active_shadows.forEach(trade => {
                const isLong = trade.direction === 'LONG';
                const distToBarycenter = Math.abs(trade.entry_price - trade.barycenter);
                const color = isLong ? 'var(--cyan)' : 'var(--red)';
                const html = `<div class="event-card" style="border-left-color: ${color}">
                    <span class="event-sym">${trade.symbol} [${trade.direction}] - Floating</span>
                    <span style="font-size:0.8rem">Risk: $${trade.risk_amount.toFixed(2)} | Entry: ${trade.entry_price.toFixed(5)}</span><br>
                    <span style="color:var(--text-main); font-size: 0.75rem;">Target: ${trade.barycenter.toFixed(5)}</span>
                </div>`;
                activeContainer.insertAdjacentHTML('beforeend', html);
            });
        }
    }

    if (data.closed_shadows !== undefined) {
        const historyContainer = document.getElementById('shadow-history');
        historyContainer.innerHTML = '';
        const reversedHistory = [...data.closed_shadows].reverse();
        reversedHistory.forEach(trade => {
            const isWin = trade.pnl > 0;
            const color = isWin ? '#45A29E' : '#FF2A2A';
            const sign = isWin ? '+' : '';
            const html = `<div class="event-card" style="border-left-color: ${color}; opacity: 0.8; padding: 0.5rem; margin-bottom: 0.4rem;">
                <span style="display:flex; justify-content:space-between; width:100%;">
                    <strong>${trade.symbol}</strong>
                    <span style="color:${color}">${sign}$${trade.pnl.toFixed(2)}</span>
                </span>
                <span style="font-size: 0.7rem; color:var(--text-main)">Exit: ${trade.exit_price.toFixed(5)} @ ${trade.exit_time}</span>
            </div>`;
            historyContainer.insertAdjacentHTML('beforeend', html);
        });
    }
}

window.onload = () => {
    initChart();
    initNetwork();
    connectWebSocket();
};
