// medicine-comparison.js

// DOM Elements
const searchForm = document.getElementById('search-form');
const medicineSearch = document.getElementById('medicine-search');
const loadingElement = document.getElementById('loading');
const drugInfoSection = document.getElementById('drug-info');
const alternativesSection = document.getElementById('alternatives-section');
const viewDetailsBtn = document.getElementById('view-details-btn');
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
const meterNeedle = document.getElementById('meterNeedle');
const meterValue = document.getElementById('meterValue');

// State management
let currentView = 'table'; // 'table' or 'graph'
let drugsData = [];
let currentDrugName = '';

// Initialize page based on which page we're on
document.addEventListener('DOMContentLoaded', () => {
    // Check which page we're on
    if (searchForm) {
        // We're on the search page
        setupSearchPage();
    } else if (viewToggle) {
        // We're on the details page
        setupDetailsPage();
    }
});

// Setup for search page
function setupSearchPage() {
    // Tab functionality
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            const tabId = tab.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Form submission handler
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const searchTerm = medicineSearch.value.trim();
        
        if (!searchTerm) {
            alert('Please enter a medicine name to search.');
            return;
        }
        
        // Show loading state
        loadingElement.style.display = 'block';
        drugInfoSection.style.display = 'none';
        alternativesSection.style.display = 'none';
        
        try {
            // Fetch data from API
            const data = await fetchDrugData(searchTerm);
            currentDrugName = searchTerm;
            
            // Update UI with the fetched data
            updateDrugInfo(data);
            
            // Hide loading and show results
            loadingElement.style.display = 'none';
            drugInfoSection.style.display = 'block';
            alternativesSection.style.display = 'block';
        } catch (error) {
            console.error('Error fetching drug data:', error);
            loadingElement.style.display = 'none';
            alert('Failed to fetch drug data. Please try again.');
        }
    });
    
    // View details button handler
    if (viewDetailsBtn) {
        viewDetailsBtn.addEventListener('click', () => {
            // Navigate to similarity details page
            window.location.href = `alternate_extra.html?drug=${encodeURIComponent(currentDrugName)}`;
        });
    }
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
                // Render the data
                renderTableView();
                renderGraphView();
            })
            .catch(error => {
                console.error('Error fetching drug data:', error);
                alert('Failed to fetch drug data. Please try again.');
            });
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Load saved view preference if exists
    loadViewPreference();
}

// Fetch drug data from API
// Fetch drug data from API
async function fetchDrugData(drugName) {
    // Use the ngrok URL instead of the local one
    const apiUrl = `https://adb2-2409-40e3-3d-ee57-3116-fa86-8e83-c758.ngrok-free.app/api/get_alternative?name=${encodeURIComponent(drugName)}`;

    try {
        const response = await fetch(apiUrl, {
            method: 'GET', // Assuming it's a GET request based on the URL structure
            headers: {
                // Add the ngrok-skip-browser-warning header
                'ngrok-skip-browser-warning': 'true'
                // Add other headers if needed by this specific API endpoint
            }
        });

        if (!response.ok) {
            // Throw an error with more details if the response status is not OK
            const errorBody = await response.text(); // Attempt to get the response body for debugging
            const errorMessage = `Failed to fetch drug data for ${drugName}: Server returned status ${response.status}.`;
            console.error(errorMessage, 'Response body:', errorBody);
            throw new Error(errorMessage);
        }

        // Return the JSON data
        return await response.json();

    } catch (error) {
        // Log the error and re-throw it so the calling code can handle it
        console.error('Error in fetchDrugData:', error);
        throw error; // Re-throw the error
    }
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
        card.className = 'alternative-card';
        card.innerHTML = `
            <span class="similarity-percentage">${Math.round(alt.similarity_percentage)}% Similar</span>
            <h3 class="card-title">${alt.name}</h3>
            <p class="card-subtitle">${alt.generic_name}</p>
            <p class="card-text">${alt.indication || 'No indication information available'}</p>
            <div class="card-details">
                <div class="detail-item">
                    <div class="detail-label">Drug ID</div>
                    <div class="detail-value">${alt.drug_id}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Mechanism of Action</div>
                    <div class="detail-value">${alt.mechanism || 'No mechanism information available'}</div>
                </div>
            </div>
            <button class="show-more-btn">Show More</button>
        `;
        
        // Add click event for the Show More button
        const showMoreBtn = card.querySelector('.show-more-btn');
        const cardDetails = card.querySelector('.card-details');
        
        showMoreBtn.addEventListener('click', () => {
            if (cardDetails.classList.contains('expanded')) {
                cardDetails.classList.remove('expanded');
                showMoreBtn.textContent = 'Show More';
            } else {
                cardDetails.classList.add('expanded');
                showMoreBtn.textContent = 'Show Less';
                
                // Show the View Similarity Details button when at least one card is expanded
                if (footerActions) {
                    footerActions.style.display = 'flex';
                }
            }
        });
        
        alternativesGrid.appendChild(card);
    });
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

// Render the table view on details page
function renderTableView() {
    if (!drugTableBody) return;
    
    drugTableBody.innerHTML = '';
    
    drugsData.forEach(drug => {
        const row = document.createElement('tr');
        const similarityPercentage = Math.round(drug.similarity_percentage);
        
        row.innerHTML = `
            <td>${drug.name}</td>
            <td>${drug.generic_name}</td>
            <td>${drug.brand_name || 'Not Available'}</td>
            <td>
                <span class="similarity-badge ${getSimilarityClass(similarityPercentage)}">
                    ${similarityPercentage}%
                </span>
            </td>
        `;
        
        // Add click event to show drug details
        row.addEventListener('click', () => {
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

// Show drug details in modal
function showDrugDetails(drug) {
    if (!modalTitle || !meterNeedle || !meterValue || !drugDetails) return;
    
    modalTitle.textContent = drug.name;
    
    // Set up similarity meter
    const percentValue = drug.similarity_percentage;
    const needleRotation = (percentValue * 1.8) - 90; // Convert percentage to degrees (0-180)
    meterNeedle.style.height = '100px';
    meterNeedle.style.transform = `rotate(${needleRotation}deg)`;
    meterValue.textContent = Math.round(percentValue) + '%';
    
    // Apply color to meter based on similarity
    meterNeedle.style.backgroundColor = getSimilarityColor(percentValue);
    
    // Populate drug details
    drugDetails.innerHTML = `
        <div class="detail-card">
            <div class="detail-title">Generic Name</div>
            <div class="detail-content">${drug.generic_name || 'Not Available'}</div>
        </div>
        <div class="detail-card">
            <div class="detail-title">Brand Name</div>
            <div class="detail-content">${drug.brand_name || 'Not Available'}</div>
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