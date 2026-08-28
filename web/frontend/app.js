document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = '/api';

  // DOM Elements
  const modelSelect = document.getElementById('model-select');
  const modelDescription = document.getElementById('model-description');
  const presetSelect = document.getElementById('preset-select');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const selectedFileCard = document.getElementById('selected-file-card');
  const selectedFileName = document.getElementById('selected-file-name');
  const selectedFileSize = document.getElementById('selected-file-size');
  const selectedFileThumb = document.getElementById('selected-file-thumb');
  const selectedFileIcon = document.getElementById('selected-file-icon');
  const changeFileBtn = document.getElementById('change-file-btn');
  const removeFileBtn = document.getElementById('remove-file-btn');
  const mcSamplesInput = document.getElementById('mc-samples');
  const mcValDisplay = document.getElementById('mc-val');
  const runBtn = document.getElementById('run-btn');

  const imgLr = document.getElementById('img-lr');
  const imgSr = document.getElementById('img-sr');
  const imgGt = document.getElementById('img-gt');
  const srPanelTag = document.getElementById('sr-panel-tag');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  const regionLabel = document.getElementById('current-region-label');
  const activeModelBadge = document.getElementById('active-model-badge');

  const metricPsnr = document.getElementById('metric-psnr');
  const metricSsim = document.getElementById('metric-ssim');
  const metricSam = document.getElementById('metric-sam');
  const metricErgas = document.getElementById('metric-ergas');

  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const historyList = document.getElementById('history-list');
  const historyBadge = document.getElementById('history-badge');
  const gpuStatus = document.getElementById('gpu-status');

  const modeBtns = document.querySelectorAll('.mode-btn');

  // Tab bar (Analyze / History) -- replaces the old slide-out drawer
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  // Upload hero -> Results section toggle elements
  const uploadHero = document.getElementById('upload-hero');
  const resultsSection = document.getElementById('results-section');
  const newAnalysisBtn = document.getElementById('new-analysis-btn');

  // State
  let currentData = null;
  let currentMode = 'sr'; // 'sr', 'uncertainty', 'ndvi'
  let selectedFile = null;
  let availableModels = [];

  async function init() {
    checkStatus();
    await loadModels();
    loadPresets();
    fetchHistory();
  }

  // ---- Tab bar switching (Analyze <-> History) ----
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`pane-${btn.dataset.tab}`).classList.add('active');
    });
  });

  // ---- Upload form <-> Results section ----
  function showResults() {
    uploadHero.classList.add('hidden');
    dropzone.classList.add('hidden');
    selectedFileCard.classList.add('hidden');
    resultsSection.classList.remove('hidden');
  }

  function showUploadForm() {
    uploadHero.classList.remove('hidden');
    if (selectedFile) {
      dropzone.classList.add('hidden');
      selectedFileCard.classList.remove('hidden');
    } else {
      dropzone.classList.remove('hidden');
      selectedFileCard.classList.add('hidden');
    }
    resultsSection.classList.add('hidden');
  }

  newAnalysisBtn.addEventListener('click', showUploadForm);

  async function checkStatus() {
    try {
      const res = await fetch(`${API_BASE}/status`);
      const data = await res.json();
      gpuStatus.textContent = `Device: ${data.gpu_name} (${data.device.toUpperCase()})`;
    } catch (e) {
      console.warn("Backend API offline or connecting...", e);
    }
  }

  // NEW: Populate the model switcher from /api/models. Models whose
  // checkpoint file isn't present on disk are shown but disabled, so a
  // teammate who hasn't copied a .pt file yet gets a clear signal instead
  // of a confusing failed request mid-demo.
  async function loadModels() {
    try {
      const res = await fetch(`${API_BASE}/models`);
      availableModels = await res.json();
      modelSelect.innerHTML = '';
      availableModels.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model.id;
        opt.textContent = model.available ? model.label : `${model.label} (checkpoint missing)`;
        opt.disabled = !model.available;
        modelSelect.appendChild(opt);
      });
      // Default to the first available model
      const firstAvailable = availableModels.find(m => m.available);
      if (firstAvailable) modelSelect.value = firstAvailable.id;
      updateModelDescription();
    } catch (e) {
      console.error("Failed to load model list", e);
    }
  }

  function updateModelDescription() {
    const model = availableModels.find(m => m.id === modelSelect.value);
    modelDescription.textContent = model ? model.description : '';
  }

  modelSelect.addEventListener('change', updateModelDescription);

  async function loadPresets() {
    try {
      const res = await fetch(`${API_BASE}/tiles`);
      const tiles = await res.json();
      presetSelect.innerHTML = '<option value="">-- Select Preset Region --</option>';
      tiles.forEach(tile => {
        const opt = document.createElement('option');
        opt.value = tile.id;
        opt.textContent = tile.name;
        presetSelect.appendChild(opt);
      });
    } catch (e) {
      console.error("Failed to load preset tiles", e);
    }
  }

  mcSamplesInput.addEventListener('input', (e) => {
    mcValDisplay.textContent = e.target.value;
  });

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function handleFileSelected(file) {
    if (!file) {
      clearFileSelection();
      return;
    }
    selectedFile = file;
    presetSelect.value = '';

    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);

    if (file.type && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        selectedFileThumb.src = e.target.result;
        selectedFileThumb.classList.remove('hidden');
        selectedFileIcon.classList.add('hidden');
      };
      reader.readAsDataURL(file);
    } else {
      selectedFileThumb.src = '';
      selectedFileThumb.classList.add('hidden');
      selectedFileIcon.classList.remove('hidden');
      selectedFileIcon.textContent = file.name.endsWith('.npy') ? '📊' : '🛰️';
    }

    dropzone.classList.add('hidden');
    selectedFileCard.classList.remove('hidden');
  }

  function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    selectedFileThumb.src = '';
    selectedFileThumb.classList.add('hidden');
    selectedFileIcon.classList.remove('hidden');
    selectedFileCard.classList.add('hidden');
    dropzone.classList.remove('hidden');
  }

  dropzone.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
      fileInput.click();
    }
  });
  fileInput.addEventListener('click', (e) => e.stopPropagation());

  changeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFileSelection();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      try {
        fileInput.files = e.dataTransfer.files;
      } catch (err) {
        // Fallback for browsers
      }
      handleFileSelected(file);
    }
  });

  presetSelect.addEventListener('change', () => {
    if (presetSelect.value) {
      clearFileSelection();
    }
  });

  runBtn.addEventListener('click', async () => {
    const tileId = presetSelect.value;
    if (!tileId && !selectedFile) {
      alert("Please select a preset satellite region or upload a custom image tile!");
      return;
    }

    const formData = new FormData();
    if (selectedFile) {
      formData.append('file', selectedFile);
    } else {
      formData.append('tile_id', tileId);
    }
    formData.append('n_samples', mcSamplesInput.value);
    formData.append('model_id', modelSelect.value);   // NEW -- tells the
    // backend which
    // of the three
    // checkpoints to run

    const modelLabel = availableModels.find(m => m.id === modelSelect.value)?.label || modelSelect.value;
    loadingText.textContent = `Running ${modelLabel} + MC-Dropout Uncertainty Analysis...`;
    loadingOverlay.classList.remove('hidden');

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || "Inference failed");
      }

      currentData = await res.json();
      renderResults(currentData);
      showResults();
      fetchHistory();
    } catch (err) {
      console.error(err);
      alert("Super-Resolution execution failed: " + err.message);
    } finally {
      loadingOverlay.classList.add('hidden');
    }
  });

  // Render all three panels -- LR, model output (SR/uncertainty/NDVI
  // depending on mode), and ground truth
  function renderResults(data) {
    const modelLabel = availableModels.find(m => m.id === data.model_id)?.label || data.model_id;
    regionLabel.textContent = `Region: ${data.name} | Model: ${modelLabel} (Scale Factor 4x)`;
    activeModelBadge.textContent = `Active model: ${modelLabel}`;

    imgLr.src = data.lr_b64;
    imgGt.src = data.gt_b64;
    updateModelOutputPanel();

    metricPsnr.textContent = data.metrics.psnr;
    metricSsim.textContent = data.metrics.ssim;
    metricSam.textContent = data.metrics.sam;
    metricErgas.textContent = data.metrics.ergas;
  }

  function updateModelOutputPanel() {
    if (!currentData) return;
    if (currentMode === 'sr') {
      imgSr.src = currentData.sr_b64;
      srPanelTag.textContent = 'Model Output';
    } else if (currentMode === 'uncertainty') {
      imgSr.src = currentData.uncertainty_b64;
      srPanelTag.textContent = 'Uncertainty Heatmap';
    } else if (currentMode === 'ndvi') {
      imgSr.src = currentData.ndvi_b64;
      srPanelTag.textContent = 'NDVI (Model Output)';
    }
  }

  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      updateModelOutputPanel();
    });
  });

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const historyItems = await res.json();
      historyBadge.textContent = historyItems.length;
      renderHistoryList(historyItems);
    } catch (e) {
      console.error("Failed to load history", e);
    }
  }

  function renderHistoryList(items) {
    if (items.length === 0) {
      historyList.innerHTML = '<p class="empty-msg">No past runs saved yet.</p>';
      return;
    }

    historyList.innerHTML = '';
    items.forEach(item => {
      const modelLabel = availableModels.find(m => m.id === item.model_id)?.label || item.model_id || '';
      const card = document.createElement('div');
      card.className = 'history-card';
      card.innerHTML = `
        <div class="history-card-header">
          <span>${item.name}</span>
          <span class="history-date">${item.date_str.split(' ')[1]}</span>
        </div>
        <div class="history-card-model">${modelLabel}</div>
        <div class="history-card-body">
          <img src="${item.sr_b64}" class="history-thumb" alt="thumb">
          <div class="history-card-metrics">
            <div>PSNR: <strong>${item.metrics.psnr || '--'}</strong> dB</div>
            <div>SSIM: <strong>${item.metrics.ssim || '--'}</strong></div>
            <div>SAM: <strong>${item.metrics.sam || '--'}</strong>°</div>
          </div>
        </div>
      `;
      card.addEventListener('click', () => {
        currentData = item;
        renderResults(item);
        showResults();
        document.getElementById('tab-btn-analyze').click();
      });
      historyList.appendChild(card);
    });
  }

  clearHistoryBtn.addEventListener('click', async () => {
    if (confirm("Are you sure you want to clear all upload history?")) {
      await fetch(`${API_BASE}/history`, { method: 'DELETE' });
      fetchHistory();
    }
  });

  init();
});