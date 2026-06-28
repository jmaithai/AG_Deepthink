const UI = {
    phi: document.getElementById('phi_val'),
    entropy: document.getElementById('entropy_val'),
    nodes: document.getElementById('nodes_val'),
    equity: document.getElementById('equity_val'),
    actionTable: document.querySelector('#actionTable tbody'),
    manifestTable: document.querySelector('#manifestTable tbody'),
    manifestSection: document.getElementById('manifestSection'),
    totalRisk: document.getElementById('totalRiskVal'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    liveToggle: document.getElementById('liveToggle'),
    modeLabel: document.getElementById('modeLabel'),
    refreshBtn: document.getElementById('refreshBtn'),
    executeBtn: document.getElementById('executeBtn')
};

// Toggle logic
UI.liveToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
        UI.modeLabel.textContent = '🔴 LIVE TRADE';
        UI.modeLabel.className = 'mode-label live';
    } else {
        UI.modeLabel.textContent = 'PAPER TRADE (DRY RUN)';
        UI.modeLabel.className = 'mode-label safe';
    }
});

// Fetch Snapshot
async function fetchSnapshot() {
    UI.loadingOverlay.classList.remove('hidden');
    UI.loadingText.textContent = "INITIALIZING MANIFOLD...";
    try {
        const res = await fetch('/api/snapshot');
        if (!res.ok) throw new Error("API Error");
        const data = await res.json();
        renderSnapshot(data);
    } catch (err) {
        console.error("Engine offline", err);
        alert("Backend offline or error fetching manifold.");
    }
    UI.loadingOverlay.classList.add('hidden');
}

function renderSnapshot(data) {
    UI.phi.textContent = data.phi.toFixed(4);
    UI.entropy.textContent = data.entropy.toFixed(4);
    UI.nodes.textContent = data.active_nodes;
    UI.equity.textContent = `$${data.equity.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

    // Render Action Table
    UI.actionTable.innerHTML = '';
    const top10 = data.actions.slice(0, 10);
    top10.forEach(a => {
        const tr = document.createElement('tr');
        const valClass = a.U_risk > 0 ? 'val-positive' : 'val-negative';
        tr.innerHTML = `
            <td>${a.Symbol}</td>
            <td class="${valClass}">${a.U_risk.toFixed(4)}</td>
            <td>${a.Abs_U.toFixed(4)}</td>
        `;
        UI.actionTable.appendChild(tr);
    });

    // Render Plotly 3D Scatter
    const x = [], y = [], z = [], text = [];
    data.nodes.forEach(n => {
        x.push(n.P); y.push(n.v); z.push(n.m); text.push(n.node);
    });

    const trace = {
        x: x, y: y, z: z, text: text,
        mode: 'markers',
        type: 'scatter3d',
        marker: { 
            size: 6, 
            color: x, 
            colorscale: 'Blues', 
            opacity: 0.8 
        }
    };
    
    const layout = {
        margin: {l: 0, r: 0, b: 0, t: 0},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            xaxis: { gridcolor: '#1a1a2e', zerolinecolor: '#00f0ff', title: 'Pressure', titlefont: {color: '#00f0ff'} },
            yaxis: { gridcolor: '#1a1a2e', zerolinecolor: '#00f0ff', title: 'Velocity', titlefont: {color: '#00f0ff'} },
            zaxis: { gridcolor: '#1a1a2e', zerolinecolor: '#00f0ff', title: 'Mass', titlefont: {color: '#00f0ff'} }
        },
        font: { color: '#00f0ff' }
    };
    
    Plotly.newPlot('manifoldChart', [trace], layout, {responsive: true, displayModeBar: false});
}

// Refresh Button
UI.refreshBtn.addEventListener('click', fetchSnapshot);

// Execute Button
UI.executeBtn.addEventListener('click', async () => {
    const isLive = UI.liveToggle.checked;
    const msg = isLive 
        ? "WARNING: You are about to place LIVE trades on MT5. Proceed?" 
        : "Run Paper Trade sequence?";
        
    if (!confirm(msg)) return;

    UI.loadingOverlay.classList.remove('hidden');
    UI.loadingText.textContent = "EXECUTING ACTION TENSOR...";
    
    try {
        const res = await fetch('/api/execute', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ dry_run: !isLive })
        });
        if (!res.ok) throw new Error("Execution failed");
        
        const result = await res.json();
        renderManifest(result);
        
        // Background refresh to update equity and state
        fetch('/api/snapshot')
            .then(r => r.json())
            .then(d => renderSnapshot(d));
            
    } catch(err) {
        console.error(err);
        alert("Execution failed.");
    }
    
    UI.loadingOverlay.classList.add('hidden');
});

function renderManifest(data) {
    UI.manifestSection.classList.remove('hidden');
    UI.manifestTable.innerHTML = '';
    
    data.manifest.forEach(row => {
        const tr = document.createElement('tr');
        let actionClass = '';
        if (row.ACTION === 'BUY') actionClass = 'val-positive';
        else if (row.ACTION === 'SELL') actionClass = 'val-negative';
        else if (row.ACTION === 'VETO') actionClass = 'val-veto';

        tr.innerHTML = `
            <td>${row.SYMBOL}</td>
            <td class="${actionClass}">${row.ACTION}</td>
            <td>${row.CONVICTION.toFixed(4)}</td>
            <td>${row.ATR.toFixed(5)}</td>
            <td>$${row["RISK ($)"].toFixed(2)}</td>
            <td>${row.LOTS.toFixed(2)}</td>
            <td>${row.STATUS}</td>
        `;
        UI.manifestTable.appendChild(tr);
    });

    UI.totalRisk.textContent = `$${data.total_risk.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
}

// Initialize on load
fetchSnapshot();
