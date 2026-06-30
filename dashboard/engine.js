// Particles JS Config
particlesJS("particles-js", {
    "particles": { "number": { "value": 40 }, "color": { "value": "#00E5FF" }, "opacity": { "value": 0.15 }, "size": { "value": 1.5 }, "line_linked": { "enable": true, "distance": 150, "color": "#00E5FF", "opacity": 0.05, "width": 1 }, "move": { "enable": true, "speed": 0.8 } }
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
        data: { labels: [], datasets: [{ label: 'Density', data: [], borderColor: '#00E5FF', backgroundColor: 'rgba(0, 229, 255, 0.15)', fill: true, tension: 0.4, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: true, ticks: { color: '#94A3B8', maxTicksLimit: 5 } }, y: { display: false } }, animation: { duration: 0 } }
    });
}

function initNetwork() {
    const container = document.getElementById('network-container');
    const data = { nodes: nodesDataset, edges: edgesDataset };
    const options = {
        nodes: { shape: 'dot', font: { color: '#8B949E', face: 'Space Grotesk', size: 10, strokeWidth: 0 }, borderWidth: 1, shadow: false },
        edges: { color: { color: 'rgba(255,255,255,0.05)', highlight: '#00E5FF' }, width: 1, smooth: { type: 'continuous' } },
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

    ws.onopen = () => { statusEl.innerText = "TELEMETRY LINK SECURE"; statusEl.className = "text-teal"; };

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
        statusEl.innerText = "LINK SEVERED. RECONNECTING..."; statusEl.className = "text-crimson";
        setTimeout(connectWebSocket, 2000);
    };
}

function updateUI(data) {
    // Top Nodes List
    const listEl = document.getElementById('top-nodes-list');
    listEl.innerHTML = '';
    let sortedNodes = [...data.nodes].sort((a, b) => Math.abs(b.sigma) - Math.abs(a.sigma));
    
    sortedNodes.slice(0, 3).forEach(node => {
        const isDanger = Math.abs(node.sigma) > 2.0;
        const colorClass = isDanger ? 'text-crimson' : 'text-primary';
        const dangerClass = isDanger ? 'danger-node' : '';
        const dir = node.force > 0 ? '↑' : '↓';
        const activeClass = selectedSymbol === node.symbol ? 'active' : '';
        
        const html = `<div class="node-item ${activeClass} ${dangerClass}" id="list-${node.symbol}" onclick="selectNode('${node.symbol}')">
            <span class="node-sym">${node.symbol}</span>
            <span class="node-sig ${colorClass}">${dir} ${node.sigma.toFixed(2)}σ</span>
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
        let nColor = '#00E5FF'; // Teal
        let nSize = 10;
        if(Math.abs(n.sigma) > 3.0) { nColor = '#FF2A55'; nSize = 25; } // Crimson
        else if(Math.abs(n.sigma) > 2.0) { nColor = '#7B2CBF'; nSize = 18; } // Violet
        
        visNodes.push({
            id: n.symbol,
            label: n.symbol,
            color: { background: '#090B10', border: nColor },
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
            const eventClass = ev.direction.includes('LONG') ? 'event-long' : 'event-short';
            const html = `<div class="event-card ${eventClass}">
                <div class="event-sym">
                    <span>${ev.symbol}</span>
                    <span class="${ev.direction.includes('LONG') ? 'text-teal' : 'text-crimson'}">${ev.direction}</span>
                </div>
                <div class="event-details">
                    <span>σ: ${ev.sigma.toFixed(2)}</span>
                    <span>RPL: ${ev.ripple_target}</span>
                    <span class="text-teal">STP: ${(ev.stop_dist*100).toFixed(2)}%</span>
                    <span class="text-teal">RSK: ${ev.risk.toFixed(1)}%</span>
                    <span class="text-dim" style="grid-column: 1 / -1">VAC: ${ev.vacuum.toFixed(5)} | FRIC: ${ev.friction.toFixed(5)}</span>
                </div>
            </div>`;
            feed.insertAdjacentHTML('afterbegin', html);
            if (feed.children.length > 4) { // 3 events + empty state hidden
                feed.lastElementChild.remove();
            }
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
                const eventClass = isLong ? 'event-long' : 'event-short';
                const distToBarycenter = Math.abs(trade.entry_price - trade.barycenter);
                const html = `<div class="event-card ${eventClass}" style="padding: 0.8rem">
                    <div class="event-sym">
                        <span>${trade.symbol}</span>
                        <span class="${isLong ? 'text-teal' : 'text-crimson'}">FLOAT</span>
                    </div>
                    <div class="event-details">
                        <span>RSK: $${trade.risk_amount.toFixed(2)}</span>
                        <span>ENT: ${trade.entry_price.toFixed(5)}</span>
                        <span class="text-dim" style="grid-column: 1 / -1">TGT: ${trade.barycenter.toFixed(5)}</span>
                    </div>
                </div>`;
                activeContainer.insertAdjacentHTML('beforeend', html);
            });
        }
    }

    if (data.closed_shadows !== undefined) {
        const historyContainer = document.getElementById('shadow-history');
        historyContainer.innerHTML = '';
        const reversedHistory = [...data.closed_shadows].reverse().slice(0, 3);
        reversedHistory.forEach(trade => {
            const isWin = trade.pnl > 0;
            const eventClass = isWin ? 'event-win' : 'event-loss';
            const colorClass = isWin ? 'text-teal' : 'text-crimson';
            const sign = isWin ? '+' : '';
            const html = `<div class="event-card ${eventClass}" style="padding: 0.6rem 0.8rem">
                <div class="event-sym" style="font-size: 0.9rem; margin-bottom: 0.2rem;">
                    <span>${trade.symbol}</span>
                    <span class="${colorClass}">${sign}$${trade.pnl.toFixed(2)}</span>
                </div>
                <div class="text-dim" style="font-size: 0.75rem;">
                    EXT: ${trade.exit_price.toFixed(5)} @ ${trade.exit_time}
                </div>
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
