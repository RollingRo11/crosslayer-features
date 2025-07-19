// ! [1/5] featureTables

function setupFeatureTables(featureIdx, componentData, containerId) {
  const tablesContainer = d3.select(`#${containerId}`).attr("class", "featureTables");

  // Add title ABOVE the tables
  if (componentData.title) {
    tablesContainer.append("h4").html(componentData.title).style("margin-bottom", "15px");
  }

  // Helper function to add a table
  function addTable(data, title) {
    // Add the table container
    const tableContainer = tablesContainer.append("div").attr("class", "table-container");

    // Add a title for this specific table
    tableContainer.append("h5").text(title).style("margin-bottom", "5px").style("font-size", "0.9em");

    // Create the table
    const table = tableContainer.append("table");

    // Create header row
    const headerRow = table.append("thead").append("tr");
    Object.keys(data[0] || {}).forEach(key => {
      headerRow.append("th").text(key).style("text-align", "left").style("padding", "5px");
    });

    // Create body rows
    const tbody = table.append("tbody");
    data.forEach(row => {
      const tr = tbody.append("tr");
      Object.entries(row).forEach(([key, value]) => {
        const td = tr.append("td").style("padding", "3px 5px");
        if (typeof value === 'number') {
          td.text(value.toFixed(3)).style("text-align", "right");
        } else {
          td.html(value);
        }
      });
    });
  }

  // Add each table from the component data
  if (componentData.neuron_alignment) {
    addTable(componentData.neuron_alignment, "Neuron Alignment");
  }
  if (componentData.correlated_neurons) {
    addTable(componentData.correlated_neurons, "Correlated Neurons");
  }
  if (componentData.correlated_features) {
    addTable(componentData.correlated_features, "Correlated Features");
  }
}

// ! [2/5] actsHistogram

function setupActsHistogram(featureIdx, componentData, containerId) {
  const histContainer = d3.select(`#${containerId}`).attr("class", "plotly-hist");

  // Create layout
  var layout = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: {
      gridcolor: "#eee",
      zerolinecolor: "#eee",
      tickvals: componentData.ticks,
      range: [0, 1.2 * Math.max(...componentData.x)],
    },
    yaxis: { gridcolor: "#eee", zerolinecolor: "#eee" },
    barmode: "relative",
    bargap: 0.01,
    showlegend: false,
    margin: { l: 50, r: 25, b: 25, t: 25, pad: 4 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    responsive: true,
    shapes: [createHistLine("black", 2, 1.0), createHistLine("black", 1, 0.9)],
    annotations: [createHistAnnotation(), createHistAnnotation()],
  };

  // Create traces
  var traces = [
    {
      x: componentData.x,
      y: componentData.y,
      type: "bar",
      marker: { color: componentData.x.map((v) => actColor(0.4 + (0.6 * v) / Math.max(...componentData.x), 1)) },
    },
  ];

  // Plot the histogram
  Plotly.newPlot(containerId, traces, layout, {
    responsive: true,
    displayModeBar: false,
    autosize: true,
    fillParent: true,
  });

  // Maybe add title
  if (componentData.title) {
    histContainer
      .node()
      .parentNode.insertBefore(document.createElement("h4"), document.getElementById(containerId)).innerHTML =
      componentData.title;
  }
}

// ! [3/5] logitsHistogram

function setupLogitsHistogram(featureIdx, componentData, containerId) {
  const histContainer = d3.select(`#${containerId}`).attr("class", "plotly-hist");

  // Add title ABOVE the graph
  if (componentData.title) {
    histContainer.append("h4").html("Logits Histogram").style("margin-bottom", "10px");
  }

  // Create layout
  var layout = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: {
      gridcolor: "#eee",
      zerolinecolor: "#eee",
      tickvals: componentData.ticks,
      range: [1.2 * Math.min(...componentData.x), 1.2 * Math.max(...componentData.x)],
    },
    yaxis: { gridcolor: "#eee", zerolinecolor: "#eee" },
    barmode: "relative",
    bargap: 0.01,
    showlegend: false,
    margin: { l: 50, r: 25, b: 25, t: 25, pad: 4 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    responsive: true,
    shapes: [createHistLine("black", 2, 1.0), createHistLine("black", 1, 0.9)],
    annotations: [createHistAnnotation(), createHistAnnotation()],
  };

  // Create traces
  traces = [
    {
      x: componentData.x.filter((value) => value >= 0),
      y: componentData.y.filter((_, i) => componentData.x[i] >= 0),
      type: "bar",
      marker: { color: "rgba(0,0,255,0.5)" },
    },
    {
      x: componentData.x.filter((value) => value < 0),
      y: componentData.y.filter((_, i) => componentData.x[i] < 0),
      type: "bar",
      marker: { color: "rgba(255,0,0,0.5)" },
    },
  ];

  // Plot the histogram
  Plotly.newPlot(containerId, traces, layout, {
    responsive: true,
    displayModeBar: false,
    autosize: true,
    fillParent: true,
  });
}

// ! [4/5] logitsTable & probeLogitsTables

function _setupLogitTables(featureIdx, componentData, containerId) {
  const logitsContainer = d3.select(`#${containerId}`).attr("class", "logitsTable");

  // Add title
  if (componentData.title) {
    logitsContainer.append("h4").html(componentData.title).style("margin-bottom", "10px");
  }

  // Helper function to create a table
  function createTable(data, className, title) {
    const tableDiv = logitsContainer.append("div").attr("class", className);
    tableDiv.append("h5").text(title).style("margin-bottom", "5px").style("font-size", "0.9em");

    const table = tableDiv.append("table");
    const tbody = table.append("tbody");

    data.forEach(row => {
      const tr = tbody.append("tr");
      tr.append("td").html(`<code>${row.token}</code>`).style("padding", "2px 8px");
      tr.append("td").text(row.value.toFixed(3)).style("padding", "2px 8px").style("text-align", "right");
    });
  }

  // Create positive and negative tables
  if (componentData.positive) {
    createTable(componentData.positive, "positive", "Top Positive");
  }
  if (componentData.negative) {
    createTable(componentData.negative, "negative", "Top Negative");
  }
}

function setupLogitTables(featureIdx, componentData, containerId) {
  _setupLogitTables(featureIdx, componentData, containerId);
}

function setupProbeLogitsTables(featureIdx, componentData, containerId) {
  _setupLogitTables(featureIdx, componentData, containerId);
}

// ! [5/5] crossLayerActivation

function setupCrossLayerActivation(featureIdx, componentData, containerId) {
  const plotContainer = d3.select(`#${containerId}`).attr("class", "plotly-line");

  // Add title ABOVE the graph
  if (componentData.title) {
    plotContainer.append("h4").html(componentData.title).style("margin-bottom", "10px");
  }

  // Create layout for line plot
  var layout = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: {
      title: componentData.xLabel || "Layer",
      gridcolor: "#eee",
      zerolinecolor: "#eee",
      tickvals: componentData.layerNumbers,
    },
    yaxis: {
      title: componentData.yLabel || "Activation Value",
      gridcolor: "#eee",
      zerolinecolor: "#eee",
    },
    margin: { l: 50, r: 20, b: 45, t: 25, pad: 4 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    responsive: true,
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: "rgba(255,255,255,0.8)",
      bordercolor: "#ccc",
      borderwidth: 1,
    },
  };

  // Create traces for mean, max, and std
  var traces = [];

  // Mean activation trace
  traces.push({
    x: componentData.layerNumbers,
    y: componentData.meanActivations,
    type: "scatter",
    mode: "lines+markers",
    name: "Mean",
    line: { color: "#1f77b4", width: 2 },
    marker: { size: 4 },
  });

  // Max activation trace
  traces.push({
    x: componentData.layerNumbers,
    y: componentData.maxActivations,
    type: "scatter",
    mode: "lines+markers",
    name: "Max",
    line: { color: "#ff7f0e", width: 2 },
    marker: { size: 4 },
  });

  // Plot the line chart
  Plotly.newPlot(containerId, traces, layout, {
    responsive: true,
    displayModeBar: false,
    autosize: true,
    fillParent: true,
  });
}

// ! [3/5] seqMultiGroup

function _setupSeqMultiGroupStandard(featureIdx, componentData, containerId) {
  const seqGroupContainer = d3.select(`#${containerId}`);

  // Add title
  if (componentData.title) {
    seqGroupContainer.append("h4").html(componentData.title);
  }

  // Process sequence groups
  componentData.sequences.forEach((seqGroup, groupIdx) => {
    const groupDiv = seqGroupContainer.append("div").attr("class", "seq-group");

    seqGroup.forEach((sequence, seqIdx) => {
      const seqDiv = groupDiv.append("div").attr("class", "seq");

      sequence.tokens.forEach((token, tokenIdx) => {
        const tokenSpan = seqDiv.append("span")
          .attr("class", "token hover-text")
          .style("background-color", actColor(sequence.activations[tokenIdx], 1))
          .text(token);

        // Add tooltip
        const tooltipContainer = seqDiv.append("div").attr("class", "tooltip-container");
        const tooltip = tooltipContainer.append("div").attr("class", "tooltip");

        tokenSpan.on("mouseenter", function(event) {
          tooltip.html(`Token: ${token}<br>Activation: ${sequence.activations[tokenIdx].toFixed(3)}`);
          tooltip.style("display", "block");

          const rect = this.getBoundingClientRect();
          tooltip.style("left", (rect.left + window.scrollX) + "px")
                 .style("top", (rect.bottom + window.scrollY + 5) + "px");
        }).on("mouseleave", function() {
          tooltip.style("display", "none");
        });
      });
    });
  });
}

function setupSeqMultiGroup(featureIdx, componentData, containerId) {
  _setupSeqMultiGroupStandard(featureIdx, componentData, containerId);
}

// ! Color and utility functions

function lossColor(loss, opacity = 1) {
  const intensity = Math.min(Math.abs(loss) * 10, 1);
  if (loss > 0) {
    return `rgba(255, ${Math.floor(255 * (1 - intensity))}, ${Math.floor(255 * (1 - intensity))}, ${opacity})`;
  } else {
    return `rgba(${Math.floor(255 * (1 - intensity))}, ${Math.floor(255 * (1 - intensity))}, 255, ${opacity})`;
  }
}

function actColor(activation, opacity = 1) {
  const intensity = Math.min(Math.abs(activation), 1);
  return `rgba(255, ${Math.floor(255 * (1 - intensity))}, 0, ${opacity})`;
}

function stringToId(str) {
  return str.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
}

function createHistLine(color, width, opacity) {
  return {
    type: 'line',
    x0: 0, y0: 0,
    x1: 1, y1: 1,
    line: { color: color, width: width },
    opacity: opacity,
    visible: false
  };
}

function createHistAnnotation() {
  return {
    x: 0.5,
    y: 0.5,
    xref: 'paper',
    yref: 'paper',
    text: '',
    showarrow: false,
    visible: false
  };
}

function measureTextWidth(text, fontSize = 14) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  context.font = `${fontSize}px monospace`;
  return context.measureText(text).width;
}

// ! Main setup functions

function updateDropdowns(keys) {
  const selectElem = d3.select('#select-0');
  if (selectElem.empty()) return;

  selectElem.selectAll('option').remove();
  keys.forEach(key => {
    selectElem.append('option').text(`Feature ${key}`).attr('value', key);
  });
}

const componentMap = {
  'featureTables': setupFeatureTables,
  'actsHistogram': setupActsHistogram,
  'logitsHistogram': setupLogitsHistogram,
  'logitTables': setupLogitTables,
  'probeLogitsTables': setupProbeLogitsTables,
  'crossLayerActivation': setupCrossLayerActivation,
  'seqMultiGroup': setupSeqMultiGroup
};

function setupPage(key) {
  const promptVisMode = Object.keys(PROMPT_DATA).length > 0;

  // Empty the contents of the grid-container
  const gridContainer = d3.select(".grid-container");
  gridContainer.selectAll("*").remove();

  if (!promptVisMode) {
    // * In feature-centric vis, keys = feature IDs, and we show that feature's data
    const feature = key;
    METADATA.layout.forEach((columnComponents, columnIdx) => {
      const columnWidth = METADATA.columnWidths[columnIdx];
      createColumn(columnComponents, columnIdx, columnWidth, feature);
    });
  } else {
    // * In prompt-centric vis, keys = stringified metrics, and we show a column for each feature
    const topFeaturesData = PROMPT_DATA[key];
    const columnComponents = METADATA.layout[0];
    const columnWidth = METADATA.columnWidths[0];
    topFeaturesData.forEach(({ feature, title }, columnIdx) => {
      createColumn(columnComponents, columnIdx, columnWidth, feature, title);
    });
  }

  function createColumn(columnComponents, columnIdx, columnWidth, feature, title = null) {
    // Insert a column into the grid-container
    const column = gridContainer
      .append("div")
      .attr("id", `column-${columnIdx}`)
      .attr("class", "grid-column")
      .attr("style", `width: ${columnWidth ? columnWidth : "auto"}px`);

    // Add an optional title, as a div
    if (title) {
      column.append("div").html(title);
    }

    // Insert each of the features' components, in order
    columnComponents.forEach((componentName, componentIdx) => {
      var t0 = performance.now();

      const componentFn = componentMap[componentName];
      if (componentFn) {
        const containerId = `${componentName}-${feature}`;
        column.append("div").attr("id", containerId).attr("class", componentName);
        componentFn(feature, DATA[feature][componentName], containerId);
      }

      var t1 = performance.now();
      console.log(`col ${columnIdx}, ${componentName}-${feature}: ${(t1 - t0).toFixed(1)} ms`);
    });
  }
}

// Enhanced window resize handler for responsive plots and layout
window.addEventListener("resize", function () {
  // Debounce resize events to avoid too many redraws
  clearTimeout(window.resizeTimeout);
  window.resizeTimeout = setTimeout(function () {
    // Find all Plotly plots and resize them
    const plotlyDivs = document.querySelectorAll('div[class*="plotly"]');
    plotlyDivs.forEach(function (div) {
      const plotDiv = div.querySelector('div[id*="plot"]') || div.children[0];
      if (plotDiv && plotDiv.id && window.Plotly && window.Plotly.Plots) {
        // Force a full relayout to handle width changes
        window.Plotly.Plots.resize(plotDiv.id).then(function () {
          // Trigger a second resize after initial one completes
          setTimeout(function () {
            window.Plotly.Plots.resize(plotDiv.id);
          }, 50);
        });
      }
    });

    // Force grid container to recalculate layout
    const gridContainer = document.querySelector(".grid-container");
    if (gridContainer) {
      gridContainer.style.display = "none";
      gridContainer.offsetHeight; // Trigger reflow
      gridContainer.style.display = "grid";
    }
  }, 50);
});

// Additional resize handler for immediate response
window.addEventListener("resize", function () {
  // Immediate resize for better responsiveness
  const plotlyDivs = document.querySelectorAll('div[class*="plotly"] > div');
  plotlyDivs.forEach(function (plotDiv) {
    if (plotDiv.id && window.Plotly && window.Plotly.Plots) {
      window.Plotly.Plots.resize(plotDiv.id);
    }
  });
});

// Enhanced observer to ensure plots fit their containers
document.addEventListener("DOMContentLoaded", function () {
  // Use ResizeObserver to monitor plot container size changes
  if (window.ResizeObserver) {
    const resizeObserver = new ResizeObserver(function (entries) {
      entries.forEach(function (entry) {
        const plotElement = entry.target.querySelector('div[id*="plot"]') || entry.target.children[0];
        if (plotElement && plotElement.id && window.Plotly && window.Plotly.Plots) {
          // Debounce the resize calls
          clearTimeout(plotElement.resizeTimeout);
          plotElement.resizeTimeout = setTimeout(function () {
            window.Plotly.Plots.resize(plotElement.id);
          }, 100);
        }
      });
    });

    // Observe all plotly containers and grid columns
    setTimeout(function () {
      const plotContainers = document.querySelectorAll(".plotly-hist, .plotly-line, .grid-column");
      plotContainers.forEach(function (container) {
        resizeObserver.observe(container);
      });
    }, 500);
  }

  // Also add a mutation observer to handle dynamic content changes
  if (window.MutationObserver) {
    const mutationObserver = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
          // Check if new Plotly plots were added
          mutation.addedNodes.forEach(function (node) {
            if (node.nodeType === 1 && node.querySelector && node.querySelector('div[id*="plot"]')) {
              setTimeout(function () {
                const plotDiv = node.querySelector('div[id*="plot"]');
                if (plotDiv && window.Plotly && window.Plotly.Plots) {
                  window.Plotly.Plots.resize(plotDiv.id);
                }
              }, 100);
            }
          });
        }
      });
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }
});
