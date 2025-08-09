// Simple visualization for crosscoder features

// Main setup function
function setupVisualization() {
    const container = document.getElementById('vis-container');
    
    // Get first feature to display
    const features = Object.keys(DATA).map(Number).sort((a, b) => a - b);
    const firstFeature = features[0];
    
    // Setup feature selector
    setupFeatureSelector(features, firstFeature);
    
    // Display feature
    displayFeature(firstFeature);
}

// Setup feature selector dropdown with density info
function setupFeatureSelector(features, currentFeature) {
    let container = document.querySelector('.feature-selector-container');
    if (!container) {
        // Create container if it doesn't exist
        const header = document.querySelector('.header');
        container = document.createElement('div');
        container.className = 'feature-selector-container';
        header.appendChild(container);
    }
    
    // Create controls HTML
    container.innerHTML = `
        <div class="selector-controls">
            <select id="feature-selector"></select>
            <div class="sort-buttons">
                <button id="sort-high" class="sort-btn" title="Sort by density (high to low)">↓</button>
                <button id="sort-low" class="sort-btn" title="Sort by density (low to high)">↑</button>
                <button id="filter-dead" class="filter-btn" title="Hide dead features">Hide Dead</button>
            </div>
        </div>
    `;
    
    const selector = document.getElementById('feature-selector');
    
    // Store original features with densities
    window.allFeatures = features.map(feat => ({
        idx: feat,
        density: FEATURE_DENSITIES[feat] || 0
    }));
    
    // Initialize with all features
    window.currentFeatures = [...window.allFeatures];
    window.hideDeadFeatures = false;
    
    updateFeatureOptions(currentFeature);
    
    // Setup event listeners
    selector.addEventListener('change', (e) => {
        displayFeature(Number(e.target.value));
    });
    
    // Sort high to low
    document.getElementById('sort-high').addEventListener('click', () => {
        window.currentFeatures.sort((a, b) => b.density - a.density);
        document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('sort-high').classList.add('active');
        updateFeatureOptions(Number(selector.value));
    });
    
    // Sort low to high
    document.getElementById('sort-low').addEventListener('click', () => {
        window.currentFeatures.sort((a, b) => a.density - b.density);
        document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('sort-low').classList.add('active');
        updateFeatureOptions(Number(selector.value));
    });
    
    // Filter dead features
    document.getElementById('filter-dead').addEventListener('click', (e) => {
        window.hideDeadFeatures = !window.hideDeadFeatures;
        e.target.textContent = window.hideDeadFeatures ? 'Show All' : 'Hide Dead';
        e.target.classList.toggle('active', window.hideDeadFeatures);
        
        if (window.hideDeadFeatures) {
            window.currentFeatures = window.allFeatures.filter(f => f.density > 0);
        } else {
            window.currentFeatures = [...window.allFeatures];
        }
        
        // Reapply current sort
        const currentSort = document.querySelector('.sort-btn.active');
        if (currentSort) {
            currentSort.click();
        } else {
            updateFeatureOptions(Number(selector.value));
        }
    });
}

// Update feature options in dropdown
function updateFeatureOptions(currentFeature) {
    const selector = document.getElementById('feature-selector');
    selector.innerHTML = '';
    
    window.currentFeatures.forEach(feat => {
        const option = document.createElement('option');
        option.value = feat.idx;
        
        // Format density with proper precision
        const densityStr = feat.density ? feat.density.toFixed(4) : '0.0000';
        option.textContent = `Feature ${feat.idx} (${densityStr}%)`;
        
        if (feat.idx === currentFeature) {
            option.selected = true;
        }
        selector.appendChild(option);
    });
    
    // Update sort button states
    document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
}

// Display a specific feature
function displayFeature(featureIdx) {
    const featureData = DATA[featureIdx];
    if (!featureData) return;
    
    // Update feature info
    updateFeatureInfo(featureIdx, featureData);
    
    // Update selected option text to show density
    const selector = document.getElementById('feature-selector');
    const selectedOption = selector.options[selector.selectedIndex];
    if (selectedOption && !selectedOption.textContent.includes('(')) {
        const density = FEATURE_DENSITIES[featureIdx] || 0;
        selectedOption.textContent = `Feature ${featureIdx} (${density.toFixed(4)}%)`;
    }
    
    // Update activation histogram
    updateHistogram(featureIdx, featureData);
    
    // Update cross-layer trajectory (feature norm over layers)
    updateCrossLayerTrajectory(featureIdx, featureData);
    
    // Update top activations
    updateTopActivations(featureIdx, featureData);
}

// Update feature info section
function updateFeatureInfo(featureIdx, featureData) {
    const infoSection = document.getElementById('feature-info');
    
    // Get histogram data for statistics
    const histData = featureData.actsHistogram || {};
    const density = FEATURE_DENSITIES[featureIdx] || 0;
    
    // Extract statistics from histogram title if available
    let stats = {
        firing_rate: (density / 100).toFixed(4),
        total_activations: 0,
        active_sequences: 0,
        mean_activation: 0,
        min_activation: 0,
        max_activation: 0
    };
    
    if (histData.title) {
        // Parse statistics from title
        const matches = {
            density: histData.title.match(/DENSITY = ([\d.]+)%/),
            nonzero: histData.title.match(/NON-ZERO ACTIVATIONS/),
            max: histData.title.match(/MAX = ([\d.]+)/),
        };
        
        if (matches.density) {
            stats.firing_rate = (parseFloat(matches.density[1]) / 100).toFixed(4);
        }
        if (matches.max) {
            stats.max_activation = parseFloat(matches.max[1]).toFixed(3);
        }
    }
    
    // Get actual data if available
    if (histData.y && histData.y.length > 0) {
        stats.total_activations = histData.y.reduce((a, b) => a + b, 0);
        stats.active_sequences = histData.y.filter(v => v > 0).length;
    }
    
    if (histData.x && histData.x.length > 0) {
        const nonZeroVals = histData.x.filter(v => v > 0);
        if (nonZeroVals.length > 0) {
            stats.min_activation = Math.min(...nonZeroVals).toFixed(3);
            stats.max_activation = Math.max(...histData.x).toFixed(3);
            stats.mean_activation = (nonZeroVals.reduce((a, b) => a + b, 0) / nonZeroVals.length).toFixed(3);
        }
    }
    
    // Calculate percentiles if we have histogram data
    let percentileHtml = '';
    if (histData.x && histData.y && histData.x.length > 0) {
        const percentiles = calculatePercentiles(histData.x, histData.y);
        percentileHtml = `
            <div class="percentiles">
                <div>50th percentile: ${percentiles[50].toFixed(3)}</div>
                <div>90th percentile: ${percentiles[90].toFixed(3)}</div>
                <div>95th percentile: ${percentiles[95].toFixed(3)}</div>
                <div>99th percentile: ${percentiles[99].toFixed(3)}</div>
            </div>
        `;
    }
    
    infoSection.innerHTML = `
        <div class="feature-header">
            <h1>${featureIdx}</h1>
        </div>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">FIRING RATE</div>
                <div class="stat-value">${stats.firing_rate}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">TOTAL ACTIVATIONS</div>
                <div class="stat-value">${stats.total_activations}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">ACTIVE SEQUENCES</div>
                <div class="stat-value">${stats.active_sequences}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">MEAN ACTIVATION</div>
                <div class="stat-value">${stats.mean_activation}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">MIN ACTIVATION</div>
                <div class="stat-value">${stats.min_activation}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">MAX ACTIVATION</div>
                <div class="stat-value">${stats.max_activation}</div>
            </div>
        </div>
        ${percentileHtml}
    `;
}

// Calculate percentiles from histogram data
function calculatePercentiles(values, counts) {
    const percentiles = {};
    const targetPercentiles = [50, 90, 95, 99];
    
    // Create cumulative distribution
    const totalCount = counts.reduce((a, b) => a + b, 0);
    let cumSum = 0;
    
    for (let p of targetPercentiles) {
        const threshold = (p / 100) * totalCount;
        cumSum = 0;
        
        for (let i = 0; i < values.length; i++) {
            cumSum += counts[i];
            if (cumSum >= threshold) {
                percentiles[p] = values[i];
                break;
            }
        }
        
        if (!(p in percentiles)) {
            percentiles[p] = values[values.length - 1];
        }
    }
    
    return percentiles;
}

// Update histogram
function updateHistogram(featureIdx, featureData) {
    const histContainer = document.getElementById('histogram-container');
    histContainer.innerHTML = '<h3>Activation Density</h3><div id="histogram-plot"></div>';
    
    const histData = featureData.actsHistogram;
    if (!histData || !histData.x || !histData.y) {
        histContainer.innerHTML += '<p style="color: #999; padding: 20px;">No histogram data available</p>';
        return;
    }
    
    // Create interactive histogram using Plotly
    const trace = {
        x: histData.x,
        y: histData.y,
        type: 'bar',
        marker: {
            color: '#6495ED',
            line: {
                color: '#4A7FC7',
                width: 0.5
            }
        },
        hovertemplate: 'Activation: %{x:.2f}<br>Count: %{y}<extra></extra>'
    };
    
    const layout = {
        margin: { t: 10, r: 30, b: 40, l: 60 },
        height: 250,
        xaxis: {
            title: 'Activation Value',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            gridcolor: '#f0f0f0'
        },
        yaxis: {
            title: 'Count',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            gridcolor: '#f0f0f0'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        hovermode: 'closest'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
    };
    
    Plotly.newPlot('histogram-plot', [trace], layout, config);
}

// Update top activations section
function updateTopActivations(featureIdx, featureData) {
    const container = document.getElementById('top-activations');
    container.innerHTML = '<h3>Top Activations</h3>';
    
    // Try to find sequence data in seqMultiGroup
    const seqMultiGroup = featureData.seqMultiGroup;
    
    if (!seqMultiGroup) {
        container.innerHTML += '<p style="color: #999; padding: 20px;">No sequence data available</p>';
        return;
    }
    
    // Parse the seqMultiGroup data structure
    let allSequences = [];
    
    // seqMultiGroup should be an array of groups
    if (Array.isArray(seqMultiGroup)) {
        seqMultiGroup.forEach(group => {
            if (group.seqGroupData && Array.isArray(group.seqGroupData)) {
                group.seqGroupData.forEach(seqData => {
                    // Each seqData contains the actual sequence information
                    if (seqData.seqData && Array.isArray(seqData.seqData)) {
                        // Find the maximum activation in this sequence
                        let maxVal = 0;
                        let maxIdx = -1;
                        
                        seqData.seqData.forEach((token, i) => {
                            if (token.featAct && token.featAct > maxVal) {
                                maxVal = token.featAct;
                                maxIdx = i;
                            }
                        });
                        
                        // Only add sequences with actual activations
                        if (maxVal > 0) {
                            allSequences.push({
                                tokens: seqData.seqData,
                                maxVal: maxVal,
                                maxIdx: maxIdx
                            });
                        }
                    }
                });
            }
        });
    }
    
    // Sort sequences by max activation value
    allSequences.sort((a, b) => b.maxVal - a.maxVal);
    
    if (allSequences.length === 0) {
        container.innerHTML += '<p style="color: #999; padding: 20px;">No sequences with activations found</p>';
        return;
    }
    
    // Display sequences
    allSequences.slice(0, 100).forEach((seq, idx) => {
        const seqDiv = document.createElement('div');
        seqDiv.className = 'activation-sequence';
        
        // Determine the window of tokens to show (centered around max activation)
        const contextSize = 12; // Number of tokens to show on each side
        const startIdx = Math.max(0, seq.maxIdx - contextSize);
        const endIdx = Math.min(seq.tokens.length, seq.maxIdx + contextSize + 1);
        
        // Build three sections: before, center (max token), and after
        let beforeHtml = '';
        let centerHtml = '';
        let afterHtml = '';
        
        // Add ellipsis at the beginning if needed
        if (startIdx > 0) {
            beforeHtml = '<span class="token ellipsis">... </span>';
        }
        
        // Process tokens before the max
        for (let i = startIdx; i < seq.maxIdx; i++) {
            const token = seq.tokens[i];
            const act = token.featAct || 0;
            const opacity = act > 0 ? Math.min(1, act / seq.maxVal) : 0;
            const tokenStr = token.tok || token.token || '';
            
            if (act > 0) {
                beforeHtml += `<span class="token" style="background-color: rgba(255, 165, 0, ${opacity * 0.3})">${escapeHtml(tokenStr)}</span>`;
            } else {
                beforeHtml += `<span class="token">${escapeHtml(tokenStr)}</span>`;
            }
        }
        
        // Add the max activation token (center)
        if (seq.maxIdx < seq.tokens.length) {
            const maxToken = seq.tokens[seq.maxIdx];
            const tokenStr = maxToken.tok || maxToken.token || '';
            centerHtml = `<span class="token max-activation">${escapeHtml(tokenStr)}</span>`;
        }
        
        // Process tokens after the max
        for (let i = seq.maxIdx + 1; i < endIdx; i++) {
            const token = seq.tokens[i];
            const act = token.featAct || 0;
            const opacity = act > 0 ? Math.min(1, act / seq.maxVal) : 0;
            const tokenStr = token.tok || token.token || '';
            
            if (act > 0) {
                afterHtml += `<span class="token" style="background-color: rgba(255, 165, 0, ${opacity * 0.3})">${escapeHtml(tokenStr)}</span>`;
            } else {
                afterHtml += `<span class="token">${escapeHtml(tokenStr)}</span>`;
            }
        }
        
        // Add ellipsis at the end if needed
        if (endIdx < seq.tokens.length) {
            afterHtml += '<span class="token ellipsis"> ...</span>';
        }
        
        // Format activation value
        const actValue = seq.maxVal.toFixed(3);
        
        seqDiv.innerHTML = `
            <div class="sequence-content">
                <div class="tokens tokens-before">${beforeHtml}</div>
                <div class="tokens tokens-center">${centerHtml}</div>
                <div class="tokens tokens-after">${afterHtml}</div>
                <div class="activation-value">${actValue}</div>
            </div>
        `;
        
        container.appendChild(seqDiv);
    });
}

// Escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Update cross-layer trajectory (feature norm over layers)
function updateCrossLayerTrajectory(featureIdx, featureData) {
    const container = document.getElementById('cross-layer-trajectory');
    container.innerHTML = '<h3>Cross-Layer Feature Trajectory</h3>';
    
    // Try to find cross-layer trajectory data - check multiple possible locations
    const trajectoryData = featureData.crossLayerTrajectory || 
                          featureData.cross_layer_trajectory ||
                          featureData.decoderNorms;
    
    if (!trajectoryData) {
        container.innerHTML += '<p style="color: #999; padding: 20px;">No trajectory data available</p>';
        return;
    }
    
    // Handle different data structures
    let layers = [];
    let norms = [];
    
    // Check if trajectoryData has the expected structure from CrossLayerTrajectoryData
    if (trajectoryData.layers && trajectoryData.trajectories) {
        // This is the expected structure from the Python code
        layers = trajectoryData.layers;
        // Use the first trajectory (should be the feature norm trajectory)
        norms = trajectoryData.trajectories[0] || trajectoryData.meanTrajectory || [];
    } else if (trajectoryData.meanTrajectory) {
        // Alternative: use meanTrajectory directly
        layers = trajectoryData.layers || Array.from({length: trajectoryData.meanTrajectory.length}, (_, i) => i);
        norms = trajectoryData.meanTrajectory;
    } else if (Array.isArray(trajectoryData)) {
        // Direct array of norms
        layers = Array.from({length: trajectoryData.length}, (_, i) => i);
        norms = trajectoryData;
    } else if (trajectoryData.data) {
        // Nested data structure - recurse
        return updateCrossLayerTrajectory(featureIdx, {crossLayerTrajectory: trajectoryData.data});
    }
    
    // Log for debugging
    console.log('Trajectory data structure:', trajectoryData);
    console.log('Extracted layers:', layers);
    console.log('Extracted norms:', norms);
    
    if (layers.length === 0 || norms.length === 0) {
        container.innerHTML += '<p style="color: #999; padding: 20px;">No valid trajectory data found</p>';
        return;
    }
    
    // Add a div for the Plotly chart
    container.innerHTML += '<div id="trajectory-plot"></div>';
    
    // Create interactive line chart using Plotly
    const trace = {
        x: layers,
        y: norms,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: '#ff6b6b',
            width: 2
        },
        marker: {
            color: '#ff6b6b',
            size: 6
        },
        hovertemplate: 'Layer: %{x}<br>Feature Norm: %{y:.3f}<extra></extra>'
    };
    
    const layout = {
        margin: { t: 10, r: 30, b: 40, l: 60 },
        height: 250,
        xaxis: {
            title: 'Layer',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            gridcolor: '#f0f0f0',
            dtick: 1
        },
        yaxis: {
            title: 'Feature Norm',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            gridcolor: '#f0f0f0'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        hovermode: 'closest'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
    };
    
    Plotly.newPlot('trajectory-plot', [trace], layout, config);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', setupVisualization);