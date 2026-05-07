// ==================== CONFIG ====================
const API_URL = window.location.origin;

// ==================== STATE ====================
let selectedFile = null;

// ==================== DOM ELEMENTS ====================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('imageFile');
const browseBtn = document.getElementById('browseBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysis');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('statusText');
const predictionBadge = document.getElementById('predictionBadge');
const confidence = document.getElementById('confidence');
const realProbBar = document.getElementById('realProbBar');
const fakeProbBar = document.getElementById('fakeProbBar');
const realProbValue = document.getElementById('realProbValue');
const fakeProbValue = document.getElementById('fakeProbValue');

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    setupEventListeners();
});

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // Upload area click (but not if clicking on browse button)
    uploadArea.addEventListener('click', (e) => {
        if (e.target.id !== 'browseBtn') {
            fileInput.click();
        }
    });
    
    // Browse button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove file button
    removeFileBtn.addEventListener('click', resetUpload);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // New analysis button
    newAnalysisBtn.addEventListener('click', resetUpload);
}

// ==================== API HEALTH CHECK ====================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            statusDot.className = 'status-dot';
            statusText.textContent = 'Model Ready';
        } else {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Model Not Loaded';
        }
    } catch (err) {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'API Offline';
        console.error('API health check failed:', err);
    }
}

// ==================== FILE HANDLING ====================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndShowFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndShowFile(file);
    }
}

function validateAndShowFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a JPG, JPEG, PNG, or GIF image.');
        return;
    }
    
    // Validate file size (50MB max)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File is too large. Maximum size is 50MB.');
        return;
    }
    
    // Store file and show info
    selectedFile = file;
    showFileInfo(file);
}

function showFileInfo(file) {
    // Hide upload area
    uploadArea.style.display = 'none';
    
    // Show file info
    fileInfo.classList.add('show');
    
    // Set file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Enable analyze button
    analyzeBtn.disabled = false;
    analyzeBtn.classList.add('active');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ==================== IMAGE ANALYSIS ====================
async function analyzeImage() {
    if (!selectedFile) return;
    
    // Hide error if showing
    error.classList.remove('show');
    
    // Show loading state
    analyzeBtn.style.display = 'none';
    loading.classList.add('show');
    
    // Create form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        // Call API
        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show results
            displayResults(data.detection);
        } else {
            throw new Error(data.error || 'Detection failed');
        }
        
    } catch (err) {
        console.error('Analysis error:', err);
        showError(err.message || 'Failed to analyze image. Please check if the API is running.');
        // Reset UI on error
        analyzeBtn.style.display = 'block';
        loading.classList.remove('show');
    }
}

// ==================== RESULTS DISPLAY ====================
function displayResults(detection) {
    console.log('Detection result:', detection); // Debug log
    
    // Hide file info and loading
    fileInfo.classList.remove('show');
    loading.classList.remove('show');
    error.classList.remove('show');
    
    // Show results section
    results.classList.add('show');
    
    // Safely extract values with defaults
    const prediction = detection.prediction || 'Unknown';
    const isReal = prediction.toLowerCase() === 'real';
    const conf = detection.confidence || 0;
    
    // Get probabilities - they come as an object {real: X, fake: Y}
    const probabilities = detection.probabilities || { real: 0, fake: 0 };
    const realProb = probabilities.real || 0;
    const fakeProb = probabilities.fake || 0;
    
    // Set verdict badge
    if (isReal) {
        predictionBadge.textContent = 'Real';
        predictionBadge.className = 'prediction-badge real';
    } else {
        predictionBadge.textContent = 'Deepfake';
        predictionBadge.className = 'prediction-badge fake';
    }
    
    // Set confidence
    confidence.textContent = conf.toFixed(2) + '%';
    
    // Set probability bars
    realProbValue.textContent = realProb.toFixed(2) + '%';
    fakeProbValue.textContent = fakeProb.toFixed(2) + '%';
    
    // Animate bars
    setTimeout(() => {
        realProbBar.style.width = realProb + '%';
        fakeProbBar.style.width = fakeProb + '%';
    }, 100);
}

// ==================== ERROR HANDLING ====================
function showError(message) {
    // Hide other sections
    uploadArea.style.display = 'none';
    fileInfo.classList.remove('show');
    loading.classList.remove('show');
    results.classList.remove('show');
    
    // Show error section
    error.classList.add('show');
    errorMessage.textContent = message;
}

// ==================== RESET ====================
function resetUpload() {
    // Reset state
    selectedFile = null;
    fileInput.value = '';
    
    // Hide all sections except upload
    fileInfo.classList.remove('show');
    loading.classList.remove('show');
    results.classList.remove('show');
    error.classList.remove('show');
    
    // Show upload area
    uploadArea.style.display = 'block';
    
    // Reset button state
    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('active');
    analyzeBtn.style.display = 'block';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
