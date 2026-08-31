// medicine-comparison.js

// DOM Elements
const searchForm = document.getElementById('search-form');
const medicineSearch = document.getElementById('medicine-search');
const loadingElement = document.getElementById('loading');
const drugInfoSection = document.getElementById('drug-info');
const alternativesSection = document.getElementById('alternatives-section');
const viewDetailsBtn = document.getElementById('view-details-btn');
const downloadBtn = document.getElementById('download-btn');
const footerActions = document.getElementById('footer-actions');
const backButton = document.getElementById('backButton');
const viewToggle = document.getElementById('viewToggle');
const tableView = document.getElementById('tableView');
const graphView = document.getElementById('graphView');
const drugTableBody = document.getElementById('drugTableBody');
const barChart = document.getElementById('barChart');
const drugModal = document.getElementById('drugModal');
const modalClose = document.getElementById('modalClose');
const modalTitle = document.getElementById('modalTitle');
const drugDetails = document.getElementById('drugDetails');
const meterValue = document.getElementById('meterValue');
const suggestionsDropdown = document.getElementById('suggestions-dropdown');

// Medicine database (sample data - in production would come from API)
const medicineDatabase = [
    { name: "Ibuprofen", type: "NSAID" },
    { name: "Paracetamol", type: "Analgesic" },
    { name: "Amoxicillin", type: "Antibiotic" },
    { name: "Omeprazole", type: "PPI" },
    { name: "Atorvastatin", type: "Statin" },
    { name: "Metformin", type: "Antidiabetic" },
    { name: "Aspirin", type: "NSAID" },
    { name: "Lisinopril", type: "ACE Inhibitor" },
    { name: "Levothyroxine", type: "Thyroid Hormone" },
    { name: "Simvastatin", type: "Statin" }
];

// State management
let currentView = 'table'; // 'table' or 'graph'
let drugsData = [];
let currentDrugName = '';

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Setup search functionality
    if (searchForm) {
        setupSearchPage();
    } 
    
    // Setup details page if needed
    if (viewToggle) {
        setupDetailsPage();
    }

    // Structure modal close handler
    const structureModalClose = document.getElementById('structureModalClose');
    if (structureModalClose) {
        structureModalClose.addEventListener('click', () => {
            document.getElementById('structureModal').classList.remove('active');
        });
    }
});

// Setup search page functionality
function setupSearchPage() {
    // Initialize search suggestions
    const searchInput = document.getElementById('medicine-search');
    const popularItems = document.querySelectorAll('.popular-item');

    // Show suggestions when input is focused
    searchInput.addEventListener('focus', () => {
        showSuggestions(medicineDatabase.slice(0, 6));
    });

    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !suggestionsDropdown.contains(e.target)) {
            suggestionsDropdown.style.display = 'none';
        }
    });

    // Handle input changes
    searchInput.addEventListener('input', (e) => {
        const value = e.target.value.toLowerCase();
        if (value.length > 0) {
            const filtered = medicineDatabase.filter(med => 
                med.name.toLowerCase().includes(value)
            ).slice(0, 6);
            showSuggestions(filtered);
        } else {
            showSuggestions(medicineDatabase.slice(0, 6));
        }
    });

    // Handle suggestion selection
    suggestionsDropdown.addEventListener('click', (e) => {
        const suggestionItem = e.target.closest('.suggestion-item');
        if (suggestionItem) {
            const suggestionName = suggestionItem.querySelector('.suggestion-name').textContent;
            searchInput.value = suggestionName;
            suggestionsDropdown.style.display = 'none';
            searchForm.dispatchEvent(new Event('submit'));
        }
    });

    // Handle popular item clicks
    popularItems.forEach(item => {
        item.addEventListener('click', () => {
            searchInput.value = item.textContent;
            searchForm.dispatchEvent(new Event('submit'));
        });
    });

    // Tab functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active', 'is-active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            tab.classList.add('active', 'is-active');
            const tabId = tab.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Form submission handler
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const searchTerm = medicineSearch.value.trim();
        
        if (!searchTerm) {
            showMessage('Please enter a medicine name to search.', 'warning');
            return;
        }
        
        clearMessage();
        loadingElement.style.display = 'block';
        drugInfoSection.style.display = 'none';
        alternativesSection.style.display = 'none';
        footerActions.style.display = 'none';
        
        try {
            const data = await fetchDrugData(searchTerm);
            currentDrugName = searchTerm;
            updateDrugInfo(data);
            
            loadingElement.style.display = 'none';
            drugInfoSection.style.display = 'block';
            alternativesSection.style.display = 'block';
            footerActions.style.display = 'flex';
        } catch (error) {
            console.error('Error fetching drug data:', error);
            loadingElement.style.display = 'none';
            // Surface the backend's own message ("not found", rate limited,
            // server asleep) rather than a generic failure string.
            showMessage(error && error.message
                ? error.message
                : 'Failed to fetch drug data. Please try again.', 'error');
        }
    });

    // Download button handler
    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            if (!currentDrugName) {
                showMessage('Search for a drug before downloading its data.', 'warning');
                return;
            }
            
            try {
                downloadBtn.textContent = 'Downloading...';
                await downloadDrugData(currentDrugName);
                downloadBtn.textContent = 'Download Data';
            } catch (error) {
                console.error('Error downloading drug data:', error);
                downloadBtn.textContent = 'Download Data';
                showMessage(error && error.message ? error.message : 'Failed to download drug data. Please try again.', 'error');
            }
        });
    }
}

// Show search suggestions
function showSuggestions(suggestions) {
    if (suggestions.length > 0) {
        suggestionsDropdown.innerHTML = suggestions.map(med => `
            <div class="suggestion-item">
                <svg class="suggestion-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <span class="suggestion-name">${med.name}</span>
                <span class="suggestion-type">${med.type}</span>
            </div>
        `).join('');
        suggestionsDropdown.style.display = 'block';
    } else {
        suggestionsDropdown.innerHTML = '<div class="suggestion-item">No results found</div>';
        suggestionsDropdown.style.display = 'block';
    }
}




// Fetch drug data from API
async function fetchDrugData(drugName) {
    return await window.MedSphereConfig.apiFetch('/api/get_alternative', { name: drugName });
}

// Download drug data
async function downloadDrugData(drugName) {
    const data = await fetchDrugData(drugName);
    
    // Create a downloadable JSON file
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${drugName}_data.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Function to update UI with drug data on search page
function updateDrugInfo(data) {
    const targetDrug = data.target;
    
    // Update drug info section
    document.getElementById('drug-name').textContent = targetDrug.name;
    document.getElementById('drug-id').textContent = targetDrug.drug_id;
    document.getElementById('drug-synonyms').textContent = targetDrug.synonyms.split('\n').join(', ');
    document.getElementById('drug-background').textContent = targetDrug.description || 'No description available';
    document.getElementById('drug-indication').textContent = targetDrug.indication || 'No indication information available';
    document.getElementById('drug-category').textContent = targetDrug.generic_name || 'No category information available';
    document.getElementById('drug-mechanism').textContent = targetDrug.mechanism || 'No mechanism information available';
    document.getElementById('drug-pharmacodynamics').textContent = targetDrug.pharmacodynamics || 'No pharmacodynamics information available';
    
    // Update alternatives section
    document.getElementById('best-alternative-content').textContent = data.recommendation || 'No recommendation available';
    
    // Generate alternative cards
    const alternativesGrid = document.getElementById('alternatives-grid');
    alternativesGrid.innerHTML = '';
    
    // Get the top 3 alternatives
    const topAlternatives = data.similar.slice(0, 3);
    
    topAlternatives.forEach(alt => {
        const card = document.createElement('div');
        card.className = 'alternative-card ms-panel';
        card.innerHTML = `
            <span class="similarity-percentage ms-badge ms-badge--accent">${Math.round(alt.similarity_percentage)}% Similar</span>
            <h3 class="card-title">${alt.name}</h3>
            <p class="card-subtitle">${alt.generic_name}</p>
            <p class="card-text">${alt.indication || 'No indication information available'}</p>
            <button class="show-more-btn ms-btn ms-btn--ghost ms-btn--sm">Show More</button>
            <div class="card-details">
                <div class="ms-kv">
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Drug ID</div>
                        <div class="detail-value ms-kv__value">${alt.drug_id}</div>
                    </div>
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Mechanism of Action</div>
                        <div class="detail-value ms-kv__value">${alt.mechanism || 'No mechanism information available'}</div>
                    </div>
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Pharmacodynamics</div>
                        <div class="detail-value ms-kv__value">${alt.pharmacodynamics || 'No pharmacodynamics information available'}</div>
                    </div>
                </div>
            </div>
        `;
        
        // Add click event for the Show More button
        const showMoreBtn = card.querySelector('.show-more-btn');
        const cardDetails = card.querySelector('.card-details');
        
        showMoreBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (cardDetails.classList.contains('expanded')) {
                cardDetails.classList.remove('expanded');
                showMoreBtn.textContent = 'Show More';
            } else {
                cardDetails.classList.add('expanded');
                showMoreBtn.textContent = 'Show Less';
            }
        });
        
        alternativesGrid.appendChild(card);
    });
    
    // View details button handler
    if (viewDetailsBtn) {
        viewDetailsBtn.addEventListener('click', () => {
            // Navigate to similarity details page
            window.location.href = `secondpage.html?drug=${encodeURIComponent(currentDrugName)}`;
        });
    }
}

/**
 * Show a themed, dismissible message instead of a browser alert().
 *
 * Browser alerts block the page and look nothing like the rest of the site;
 * they also encouraged throwing away the backend's own error text, which is
 * more useful than a generic "failed" string (e.g. "Medicine 'xyx' not found.
 * Please check the spelling or try a different name.").
 *
 * @param {string} message
 * @param {"error"|"warning"|"success"} [kind="error"]
 */
function showMessage(message, kind) {
    var variant = kind || 'error';
    var slot = document.getElementById('page-message');

    if (!slot) {
        slot = document.createElement('div');
        slot.id = 'page-message';
        var host = document.querySelector('.ms-shell') || document.querySelector('main') || document.body;
        var anchor = host.querySelector('.search-section, .ms-panel') || host.firstChild;
        if (anchor && anchor.parentNode === host) {
            host.insertBefore(slot, anchor.nextSibling);
        } else {
            host.insertBefore(slot, host.firstChild);
        }
    }

    slot.className = 'ms-alert ms-alert--' + variant;
    slot.style.margin = '1.3rem 0';
    slot.setAttribute('role', variant === 'error' ? 'alert' : 'status');
    slot.textContent = message;
    slot.hidden = false;

    if (typeof slot.scrollIntoView === 'function') {
        slot.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/** Hide any message currently shown. */
function clearMessage() {
    var slot = document.getElementById('page-message');
    if (slot) slot.hidden = true;
}

/** Backwards-compatible alias for the themed message helper. */
function showDetailsError(message) {
    showMessage(message, 'error');
}

// Setup for details page
function setupDetailsPage() {
    // Get drug name from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const drugName = urlParams.get('drug');
    
    if (drugName) {
        currentDrugName = drugName;
        // Fetch data for this drug
        fetchDrugData(drugName)
            .then(data => {
                drugsData = data.similar;

                // Name the drug these alternatives belong to. Use the resolved
                // target name rather than the raw query, so arriving here from
                // a brand name ("Ziagen") shows the drug it maps to
                // ("Abacavir") instead of echoing the search term back.
                const resolvedName = (data.target && data.target.name) || drugName;
                const heading = document.getElementById('alternatives-title');
                if (heading) {
                    heading.textContent = 'Alternatives to ' + resolvedName;
                }
                document.title = 'Alternatives to ' + resolvedName + ' | MedSphere';

                // Render the data
                renderTableView();
                renderGraphView();
            })
            .catch(error => {
                console.error('Error fetching drug data:', error);
                showDetailsError(error && error.message
                    ? error.message
                    : 'Failed to fetch drug data. Please try again.');
            });
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Load saved view preference if exists
    loadViewPreference();
}

// Load molecular structure
function loadStructure(smiles) {
  const encoded = encodeURIComponent(smiles.trim());
  const url = `https://cactus.nci.nih.gov/chemical/structure/${encoded}/image`;

  const img = document.getElementById("structureImage");
  const structureModal = document.getElementById("structureModal");
  
  img.src = url;
  img.onerror = function() {
    img.alt = "Unable to load structure";
  };
  
  structureModal.classList.add('active');
}

// Show molecule structure
function showMoleculeStructure(smiles, name) {
  if (smiles) {
    loadStructure(smiles);
  } else {
    showMessage('No chemical structure (SMILES) is available for this drug.', 'warning');
  }
}

// Get similarity class for styling badges
function getSimilarityClass(percentage) {
    if (percentage >= 80) {
        return 'high-similarity';
    } else if (percentage >= 50) {
        return 'medium-similarity';
    } else {
        return 'low-similarity';
    }
}

// Get color for similarity meter
function getSimilarityColor(percentage) {
    if (percentage >= 80) {
        return '#69b578'; // Green for high similarity
    } else if (percentage >= 50) {
        return '#f39c12'; // Orange for medium similarity
    } else {
        return '#e74c3c'; // Red for low similarity
    }
}

// Save the current view preference to localStorage
function saveViewPreference() {
    localStorage.setItem('drugSimilarityView', currentView);
}

// Load the saved view preference from localStorage
function loadViewPreference() {
    const savedView = localStorage.getItem('drugSimilarityView');
    if (savedView) {
        if (savedView !== currentView) {
            currentView = savedView;
            toggleView();
        }
    }
}

// Set up all event listeners for details page
function setupEventListeners() {
    // Toggle view switch
    if (viewToggle) {
        viewToggle.addEventListener('click', toggleView);
    }
    
    // Modal close button
    if (modalClose) {
        modalClose.addEventListener('click', () => {
            drugModal.classList.remove('active');
        });
    }
    
    // Close modal when clicking outside of it
    if (drugModal) {
        drugModal.addEventListener('click', (e) => {
            if (e.target === drugModal) {
                drugModal.classList.remove('active');
            }
        });
    }
    
    // Back button
    if (backButton) {
        backButton.addEventListener('click', () => {
            // Save current view preference before navigating
            saveViewPreference();
            
            // Navigate back to main page
            window.location.href = 'firstpage.html';
        });
    }
}

function renderTableView() {
    if (!drugTableBody) return;
    
    drugTableBody.innerHTML = '';
    
    drugsData.forEach(drug => {
        const row = document.createElement('tr');
        const similarityPercentage = Math.round(drug.similarity_percentage);
        
        row.innerHTML = `
            <td>${drug.name}</td>
            <td>${drug.generic_name}</td>
            <td class="smiles-cell">${drug.smiles || 'Not Available'}</td>
            <td>
                <span class="similarity-badge ${getSimilarityClass(similarityPercentage)}">
                    ${similarityPercentage}%
                </span>
            </td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn view-structure-btn ms-btn ms-btn--ghost ms-btn--sm" data-smiles="${drug.smiles || ''}">View Structure</button>
                    <button class="action-btn view-details-btn ms-btn ms-btn--sm">View Details</button>
                </div>
            </td>
        `;
        
        // Add click event to show drug details
        row.addEventListener('click', (e) => {
            if (!e.target.classList.contains('action-btn')) {
                showDrugDetails(drug);
            }
        });
        
        // Add click event for View Structure button
        const viewStructureBtn = row.querySelector('.view-structure-btn');
        viewStructureBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (drug.smiles) {
                showMoleculeStructure(drug.smiles, drug.name);
            } else {
                showMessage('No chemical structure (SMILES) is available for this drug.', 'warning');
            }
        });
        
        // Add click event for View Details button
        const viewDetailsBtn = row.querySelector('.view-details-btn');
        viewDetailsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showDrugDetails(drug);
        });
        
        drugTableBody.appendChild(row);
    });
}

// Render the graph view on details page
function renderGraphView() {
    if (!barChart) return;
    
    barChart.innerHTML = '';
    
    // Sort drugs by percentage for better visualization
    const sortedDrugs = [...drugsData].sort((a, b) => {
        return a.similarity_percentage - b.similarity_percentage;
    });
    
    sortedDrugs.forEach(drug => {
        const percentValue = drug.similarity_percentage;
        const barHeight = (percentValue * 3) + 'px';
        
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.height = barHeight;
        
        const barLabel = document.createElement('div');
        barLabel.className = 'bar-label';
        barLabel.textContent = drug.name;
        
        const barValue = document.createElement('div');
        barValue.className = 'bar-value';
        barValue.textContent = Math.round(percentValue) + '%';
        
        bar.appendChild(barValue);
        bar.appendChild(barLabel);
        
        // Add click event to show drug details
        bar.addEventListener('click', () => {
            showDrugDetails(drug);
        });
        
        barChart.appendChild(bar);
    });
}

// Toggle between table and graph views
function toggleView() {
    if (currentView === 'table') {
        currentView = 'graph';
        viewToggle.classList.remove('table');
        viewToggle.classList.add('graph');
        tableView.style.display = 'none';
        graphView.style.display = 'block';
    } else {
        currentView = 'table';
        viewToggle.classList.remove('graph');
        viewToggle.classList.add('table');
        tableView.style.display = 'table';
        graphView.style.display = 'none';
    }
    
    // Save the current view preference
    saveViewPreference();
}

function showDrugDetails(drug) {
    if (!modalTitle || !meterValue || !drugDetails) return;
    
    modalTitle.textContent = drug.name;
    
    // Set up circular progress meter
    const percentValue = drug.similarity_percentage;
    const progressCircle = document.querySelector('.progress-ring-progress');
    const radius = progressCircle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentValue / 100) * circumference;
    
    progressCircle.style.strokeDasharray = `${circumference} ${circumference}`;
    progressCircle.style.strokeDashoffset = offset;
    
    // Apply color to meter based on similarity
    progressCircle.style.stroke = getSimilarityColor(percentValue);
    meterValue.textContent = Math.round(percentValue) + '%';
    
    // Populate drug details
    drugDetails.innerHTML = `
        <div class="detail-card">
            <div class="detail-title">Generic Name</div>
            <div class="detail-content">${drug.generic_name || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">SMILES</div>
            <div class="detail-content">${drug.smiles || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">Drug ID</div>
            <div class="detail-content">${drug.drug_id || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">Mechanism of Action</div>
            <div class="detail-content">${drug.mechanism || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">Indication</div>
            <div class="detail-content">${drug.indication || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">Synonyms</div>
            <div class="detail-content">${drug.synonyms ? drug.synonyms.split('\n').join(', ') : 'Not Available'}</div>
        </div>
    `;
    
    // Show the modal with animation
    drugModal.classList.add('active');
}