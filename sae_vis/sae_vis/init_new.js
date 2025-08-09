// New visualization initialization for crosscoder features

// Main initialization function
function initializeVisualization() {
    // Get the selected feature
    const selectedFeature = parseInt(START_KEY);
    
    // Create the main layout structure
    createLayout();
    
    // Populate with data for selected feature
    updateFeatureDisplay(selectedFeature);
    
    // Setup dropdown and navigation
    setupNavigation();
}

// Create the main layout structure
function createLayout() {
    const container = d3.select('.grid-container');
    container.selectAll('*').remove();
    
    // Create header section
    const header = container.append('div')
        .attr('class', 'feature-header');
    
    // Create main content area with two columns
    const mainContent = container.append('div')
        .attr('class', 'main-content');
    
    // Left column for stats and visualizations
    const leftColumn = mainContent.append('div')
        .attr('class', 'left-column');
    
    // Right column for sequences
    const rightColumn = mainContent.append('div')
        .attr('class', 'right-column');
    
    // Add sections to left column
    leftColumn.append('div').attr('id', 'stats-section');
    leftColumn.append('div').attr('id', 'histogram-section');
    leftColumn.append('div').attr('id', 'layer-norms-section');
    
    // Add sequences section to right column
    rightColumn.append('div').attr('id', 'sequences-section');
}

// Update display for a specific feature
function updateFeatureDisplay(featureIdx) {
    const featureData = DATA[featureIdx];
    if (!featureData) {
        console.error(`No data for feature ${featureIdx}`);
        return;
    }
    
    // Update header
    updateHeader(featureIdx, featureData);
    
    // Update visualizations
    updateHistogram(featureIdx, featureData);
    updateLayerNorms(featureIdx, featureData);
    
    // Update sequences with new highlighting behavior
    updateSequences(featureIdx, featureData);
}

// Update header with feature info
function updateHeader(featureIdx, featureData) {
    const header = d3.select('.feature-header');
    header.selectAll('*').remove();
    
    // Get density from FEATURE_DENSITIES or calculate from histogram title
    const density = FEATURE_DENSITIES[featureIdx] || 0.0;
    
    // Count active sequences from the data
    let activeSeqs = 0;
    if (featureData.seqMultiGroup && featureData.seqMultiGroup.seqGroups) {
        featureData.seqMultiGroup.seqGroups.forEach(group => {
            activeSeqs += group.data.length;
        });
    }
    
    // Create header content
    const headerContent = header.append('div')
        .attr('class', 'header-content');
    
    // Title
    headerContent.append('h2')
        .text(`Feature ${featureIdx}`);
    
    // Stats row
    const statsRow = headerContent.append('div')
        .attr('class', 'header-stats');
    
    // Firing rate
    statsRow.append('div')
        .attr('class', 'stat-item')
        .html(`<span class="stat-label">Firing Rate:</span> <span class="stat-value">${density.toFixed(3)}%</span>`);
    
    // Active sequences
    statsRow.append('div')
        .attr('class', 'stat-item')
        .html(`<span class="stat-label">Active Sequences:</span> <span class="stat-value">${activeSeqs}</span>`);
}

// Update histogram with improved styling
function updateHistogram(featureIdx, featureData) {
    const containerId = 'histogram-section';
    const container = d3.select(`#${containerId}`);
    container.selectAll('*').remove();
    
    if (!featureData.actsHistogram) return;
    
    const componentData = featureData.actsHistogram;
    
    // Create histogram container
    const histContainer = container.append('div')
        .attr('class', 'histogram-container');
    
    // Create layout with blue color scheme
    var layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            gridcolor: '#e1e4e8',
            zerolinecolor: '#e1e4e8',
            tickvals: componentData.ticks,
            range: [0, 1.2 * Math.max(...componentData.x)],
            title: 'Activation Value',
            titlefont: { size: 12, color: '#586069' },
            tickfont: { size: 10, color: '#586069' }
        },
        yaxis: {
            gridcolor: '#e1e4e8',
            zerolinecolor: '#e1e4e8',
            title: 'Count',
            titlefont: { size: 12, color: '#586069' },
            tickfont: { size: 10, color: '#586069' }
        },
        barmode: 'relative',
        bargap: 0.02,
        showlegend: false,
        margin: {l: 50, r: 20, b: 40, t: 20, pad: 4},
        height: 200,
        autosize: true
    };
    
    // Create traces with blue gradient
    const maxVal = Math.max(...componentData.x);
    var traces = [{
        x: componentData.x,
        y: componentData.y,
        type: 'bar',
        marker: {
            color: componentData.x.map(v => {
                const intensity = 0.3 + 0.7 * (v / maxVal);
                return `rgba(30, 136, 229, ${intensity})`;  // Blue gradient
            })
        },
        hovertemplate: 'Activation: %{x:.3f}<br>Count: %{y}<extra></extra>'
    }];
    
    // Plot the histogram
    Plotly.newPlot(histContainer.node(), traces, layout, {
        responsive: true,
        displayModeBar: false
    });
}

// Update layer norms visualization
function updateLayerNorms(featureIdx, featureData) {
    const containerId = 'layer-norms-section';
    const container = d3.select(`#${containerId}`);
    container.selectAll('*').remove();
    
    // Check if we have decoder norms data
    if (!featureData.crossLayerDecoderNorms) return;
    
    const componentData = featureData.crossLayerDecoderNorms;
    
    // Create plot container
    const plotContainer = container.append('div')
        .attr('class', 'layer-norms-container');
    
    // Extract decoder norms for this feature
    const decoderNorms = componentData.decoder_norms[featureIdx] || [];
    const layers = Array.from({length: componentData.n_layers}, (_, i) => i);
    
    // Create layout
    var layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            title: 'Layer',
            gridcolor: '#e1e4e8',
            zerolinecolor: '#e1e4e8',
            tickmode: 'linear',
            tick0: 0,
            dtick: Math.ceil(layers.length / 10),
            titlefont: { size: 12, color: '#586069' },
            tickfont: { size: 10, color: '#586069' }
        },
        yaxis: {
            title: 'Decoder Norm',
            gridcolor: '#e1e4e8',
            zerolinecolor: '#e1e4e8',
            titlefont: { size: 12, color: '#586069' },
            tickfont: { size: 10, color: '#586069' }
        },
        showlegend: false,
        margin: {l: 50, r: 20, b: 40, t: 10, pad: 4},
        height: 150,
        autosize: true,
        hovermode: 'x unified'
    };
    
    // Create trace for decoder norms
    var traces = [{
        x: layers,
        y: decoderNorms,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: 'rgb(34, 139, 34)',  // Green color as requested
            width: 2
        },
        marker: {
            color: 'rgb(34, 139, 34)',
            size: 4
        },
        hovertemplate: 'Layer: %{x}<br>Norm: %{y:.3f}<extra></extra>'
    }];
    
    // Plot the layer norms
    Plotly.newPlot(plotContainer.node(), traces, layout, {
        responsive: true,
        displayModeBar: false
    });
}

// Update sequences with new highlighting behavior
function updateSequences(featureIdx, featureData) {
    const container = d3.select('#sequences-section');
    container.selectAll('*').remove();
    
    if (!featureData.seqMultiGroup) return;
    
    const seqData = featureData.seqMultiGroup;
    
    // Add title
    container.append('h3')
        .text('Top Activations')
        .style('margin-bottom', '20px');
    
    // Process each sequence group
    seqData.seqGroups.forEach((group, groupIdx) => {
        const groupContainer = container.append('div')
            .attr('class', 'sequence-group');
        
        // Add group header if needed
        if (group.title) {
            groupContainer.append('h4')
                .attr('class', 'group-title')
                .text(group.title);
        }
        
        // Process each sequence in the group
        group.data.forEach((seq, seqIdx) => {
            const seqContainer = groupContainer.append('div')
                .attr('class', 'sequence-item')
                .attr('data-seq-id', `${groupIdx}-${seqIdx}`);
            
            // Find the maximum activation value and its index
            let maxAct = 0;
            let maxActIdx = -1;
            seq.tokenActivations.forEach((act, idx) => {
                if (act > maxAct) {
                    maxAct = act;
                    maxActIdx = idx;
                }
            });
            
            // Create tokens with special highlighting
            seq.tokens.forEach((token, tokenIdx) => {
                const isMaxToken = (tokenIdx === maxActIdx);
                const activation = seq.tokenActivations[tokenIdx];
                
                // Calculate background color based on activation
                const bgColor = getActivationColor(activation, isMaxToken, false);
                
                const tokenSpan = seqContainer.append('span')
                    .attr('class', 'token')
                    .attr('data-token-idx', tokenIdx)
                    .style('background-color', bgColor)
                    .style('position', 'relative')
                    .text(token);
                
                // Add hover data
                if (seq.hoverData && seq.hoverData[tokenIdx]) {
                    tokenSpan.attr('title', formatHoverData(seq.hoverData[tokenIdx]));
                }
            });
            
            // Add hover behavior to show all activations
            seqContainer
                .on('mouseenter', function() {
                    // Show all activation levels on hover
                    d3.select(this).selectAll('.token').each(function(d, idx) {
                        const token = d3.select(this);
                        const tokenIdx = parseInt(token.attr('data-token-idx'));
                        const activation = seq.tokenActivations[tokenIdx];
                        const isMaxToken = (tokenIdx === maxActIdx);
                        const bgColor = getActivationColor(activation, isMaxToken, true);
                        token.style('background-color', bgColor);
                    });
                })
                .on('mouseleave', function() {
                    // Reset to showing only max token highlighted
                    d3.select(this).selectAll('.token').each(function(d, idx) {
                        const token = d3.select(this);
                        const tokenIdx = parseInt(token.attr('data-token-idx'));
                        const activation = seq.tokenActivations[tokenIdx];
                        const isMaxToken = (tokenIdx === maxActIdx);
                        const bgColor = getActivationColor(activation, isMaxToken, false);
                        token.style('background-color', bgColor);
                    });
                });
        });
    });
}

// Helper function to get activation color
function getActivationColor(activation, isMaxToken, showAll) {
    if (!showAll && !isMaxToken) {
        // When not hovering, only highlight max token
        return 'rgba(255, 255, 255, 0)';
    }
    
    // Normalize activation (assuming max activation is around 1.0, adjust as needed)
    const normalizedAct = Math.min(activation, 1.0);
    
    if (isMaxToken) {
        // Max token gets strong highlight
        const intensity = 0.7 + 0.3 * normalizedAct;
        return `rgba(255, 235, 59, ${intensity})`;  // Yellow highlight
    } else if (showAll) {
        // Other tokens get gradient based on activation
        const intensity = 0.2 + 0.6 * normalizedAct;
        return `rgba(255, 235, 59, ${intensity})`;
    }
    
    return 'rgba(255, 255, 255, 0)';
}

// Format hover data for display
function formatHoverData(hoverData) {
    let parts = [];
    if (hoverData.activation !== undefined) {
        parts.push(`Activation: ${hoverData.activation.toFixed(3)}`);
    }
    if (hoverData.loss !== undefined) {
        parts.push(`Loss: ${hoverData.loss.toFixed(3)}`);
    }
    return parts.join('\n');
}

// Setup navigation and dropdown
function setupNavigation() {
    const dropdownContainer = d3.select('#dropdown-container');
    dropdownContainer.selectAll('*').remove();
    
    // Create feature selector
    const selectorDiv = dropdownContainer.append('div')
        .attr('class', 'feature-selector');
    
    selectorDiv.append('label')
        .text('Feature: ')
        .style('margin-right', '10px');
    
    const select = selectorDiv.append('div')
        .attr('class', 'select')
        .append('select')
        .attr('id', 'feature-select');
    
    // Add options for each feature
    const features = Object.keys(DATA).map(k => parseInt(k)).sort((a, b) => a - b);
    
    select.selectAll('option')
        .data(features)
        .enter()
        .append('option')
        .attr('value', d => d)
        .text(d => `Feature ${d}`)
        .property('selected', d => d === parseInt(START_KEY));
    
    // Add change handler
    select.on('change', function() {
        const selectedFeature = parseInt(this.value);
        updateFeatureDisplay(selectedFeature);
    });
    
    // Add navigation buttons
    const navButtons = dropdownContainer.append('div')
        .attr('class', 'nav-buttons')
        .style('margin-left', '20px');
    
    navButtons.append('button')
        .text('← Previous')
        .on('click', () => navigateFeature(-1));
    
    navButtons.append('button')
        .text('Next →')
        .on('click', () => navigateFeature(1));
}

// Navigate to next/previous feature
function navigateFeature(direction) {
    const select = document.getElementById('feature-select');
    const currentIdx = select.selectedIndex;
    const newIdx = currentIdx + direction;
    
    if (newIdx >= 0 && newIdx < select.options.length) {
        select.selectedIndex = newIdx;
        const event = new Event('change');
        select.dispatchEvent(event);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initializeVisualization);