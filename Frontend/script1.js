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

const esc = (value) => window.MedSphereConfig.escapeHtml(value);
const isMissing = (value) =>
    value === undefined || value === null || value === '' || value === 'Not Available';

// State management
let currentView = 'table'; // 'table' or 'graph'
let drugsData = [];
let currentDrugName = '';
let lastResult = null;

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
    const searchInput = document.getElementById('medicine-search');
    const popularItems = document.querySelectorAll('.popular-item');

    // Suggestions come from the API's full name index (names, synonyms and
    // brands for 17k drugs), replacing a hardcoded list of ten.
    window.MedSphereConfig.attachAutocomplete(searchInput, (name) => {
        searchInput.value = name;
        searchForm.requestSubmit ? searchForm.requestSubmit() : searchForm.dispatchEvent(new Event('submit'));
    });

    popularItems.forEach(item => {
        item.addEventListener('click', () => {
            searchInput.value = item.textContent;
            searchForm.dispatchEvent(new Event('submit'));
        });
    });

    if (viewDetailsBtn) {
        viewDetailsBtn.addEventListener('click', () => {
            const params = new URLSearchParams({ drug: currentDrugName });
            if (!approvedOnly()) params.set('all', '1');
            window.location.href = 'secondpage.html?' + params.toString();
        });
    }

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
            lastResult = data;
            currentDrugName = (data.target && data.target.name) || searchTerm;
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
                : 'Failed to fetch drug data. Please try again.',
                error && error.status === 404 ? 'warning' : 'error');
            window.MedSphereConfig.renderSuggestions(
                document.getElementById('page-message'), error, (name) => {
                    medicineSearch.value = name;
                    searchForm.dispatchEvent(new Event('submit'));
                });
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
                await downloadDrugData(currentDrugName, lastResult);
                downloadBtn.textContent = 'Download Data';
            } catch (error) {
                console.error('Error downloading drug data:', error);
                downloadBtn.textContent = 'Download Data';
                showMessage(error && error.message ? error.message : 'Failed to download drug data. Please try again.', 'error');
            }
        });
    }
}

// Fetch drug data from API
function approvedOnly() {
    const box = document.getElementById('include-unapproved');
    return !(box && box.checked);
}

// withLlm=false is used by the details page: it only needs the list, so it
// should not spend another Gemini call.
async function fetchDrugData(drugName, withLlm = true, onlyApproved = approvedOnly()) {
    return await window.MedSphereConfig.apiFetch('/api/get_alternative', {
        name: drugName,
        llm: withLlm ? '1' : '0',
        approved_only: onlyApproved ? '1' : '0'
    });
}

// Download drug data
async function downloadDrugData(drugName, data) {
    // The result is already on screen; fetching again would cost a request.
    if (!data) data = await fetchDrugData(drugName);
    
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
    const text = (value, fallback) => (isMissing(value) ? fallback : value);

    document.getElementById('drug-name').textContent = targetDrug.name;
    document.getElementById('drug-id').textContent = targetDrug.drug_id;
    document.getElementById('drug-synonyms').textContent =
        isMissing(targetDrug.synonyms) ? '' : targetDrug.synonyms.split('\n').join(', ');
    // DrugBank's "description" is missing for half the drugs; "background"
    // is the longer prose field and is filled more often.
    document.getElementById('drug-background').textContent =
        text(targetDrug.background, text(targetDrug.description, 'No background information available'));
    document.getElementById('drug-indication').textContent =
        text(targetDrug.indication, 'No indication information available');
    document.getElementById('drug-category').textContent = isMissing(targetDrug.drug_categories)
        ? 'No category information available'
        : targetDrug.drug_categories.split('\n').slice(0, 12).join(' · ');
    document.getElementById('drug-mechanism').textContent =
        text(splitMechanism(targetDrug.mechanism).prose, 'No mechanism information available');
    document.getElementById('drug-pharmacodynamics').textContent =
        text(targetDrug.pharmacodynamics, 'No pharmacodynamics information available');

    const disclaimer = document.getElementById('disclaimer-text');
    if (disclaimer) disclaimer.textContent = data.disclaimer || window.MedSphereConfig.disclaimer;

    document.getElementById('best-alternative-content').textContent =
        data.recommendation || 'No recommendation available';

    const alternativesGrid = document.getElementById('alternatives-grid');
    alternativesGrid.innerHTML = '';

    data.similar.slice(0, 3).forEach(alt => {
        const card = document.createElement('div');
        card.className = 'alternative-card ms-panel';
        card.innerHTML = `
            <span class="similarity-percentage ms-badge ms-badge--accent">${Math.round(alt.similarity_percentage)}% Similar</span>
            <h3 class="card-title">${esc(alt.name)}${alt.approved ? '' : '<span class="ms-tag-unapproved">not approved</span>'}</h3>
            <p class="card-subtitle">${esc(alt.generic_name)}</p>
            <p class="card-text">${esc(text(alt.indication, 'No indication information available'))}</p>
            <button class="show-more-btn ms-btn ms-btn--ghost ms-btn--sm">Show More</button>
            <div class="card-details">
                <div class="ms-kv">
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Drug ID</div>
                        <div class="detail-value ms-kv__value">${esc(alt.drug_id)}</div>
                    </div>
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Mechanism of Action</div>
                        <div class="detail-value ms-kv__value">${esc(text(splitMechanism(alt.mechanism).prose, 'No mechanism information available'))}</div>
                    </div>
                    <div class="detail-item ms-kv__item">
                        <div class="detail-label ms-kv__label">Pharmacodynamics</div>
                        <div class="detail-value ms-kv__value">${esc(text(alt.pharmacodynamics, 'No pharmacodynamics information available'))}</div>
                    </div>
                </div>
            </div>
        `;

        const showMoreBtn = card.querySelector('.show-more-btn');
        const cardDetails = card.querySelector('.card-details');
        showMoreBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const open = cardDetails.classList.toggle('expanded');
            showMoreBtn.textContent = open ? 'Show Less' : 'Show More';
        });

        alternativesGrid.appendChild(card);
    });
}

/**
 * DrugBank's mechanism text ends with a flattened table:
 *   TARGET ACTIONS ORGANISM
 *   AProstaglandin G/H synthase 2 inhibitor Humans
 * The leading A/U is a flag (A = known action, U = unknown), not part of the
 * name. Returns {prose, targets: [{text, known}]}.
 */
function splitMechanism(value) {
    if (isMissing(value)) return { prose: '', targets: [] };
    const parts = String(value).split(/\r?\n?TARGET ACTIONS ORGANISM\r?\n?/);
    const targets = (parts[1] || '').split(/\r?\n/)
        .map(line => line.trim())
        .filter(line => line.length > 1 && (line[0] === 'A' || line[0] === 'U'))
        .map(line => ({ text: line.slice(1).trim(), known: line[0] === 'A' }));
    return { prose: parts[0].trim(), targets };
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
        // llm=0: this page only lists alternatives, so skip the Gemini call.
        fetchDrugData(drugName, false, urlParams.get('all') !== '1')
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
            <td>${esc(drug.name)}${drug.approved ? '' : '<span class="ms-tag-unapproved">not approved</span>'}</td>
            <td>${esc(drug.generic_name)}</td>
            <td class="smiles-cell">${esc(drug.smiles || 'Not Available')}</td>
            <td>
                <span class="similarity-badge ${getSimilarityClass(similarityPercentage)}">
                    ${similarityPercentage}%
                </span>
            </td>
            <td>
                <div class="action-buttons">
                    <button class="action-btn view-structure-btn ms-btn ms-btn--ghost ms-btn--sm" data-smiles="${esc(drug.smiles || '')}">View Structure</button>
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
    const card = (title, value) => `
        <div class="detail-card">
            <div class="detail-title">${esc(title)}</div>
            <div class="detail-content">${esc(isMissing(value) ? 'Not Available' : value)}</div>
        </div>`;
    drugDetails.innerHTML = [
        card('Generic Name', drug.generic_name),
        card('SMILES', drug.smiles),
        card('Drug ID', drug.drug_id),
        card('Mechanism of Action', splitMechanism(drug.mechanism).prose),
        card('Indication', drug.indication),
        card('Synonyms', isMissing(drug.synonyms) ? '' : drug.synonyms.split('\n').join(', '))
    ].join('');

    // Show the modal with animation
    drugModal.classList.add('active');
}