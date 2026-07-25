document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommender-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    
    const recommendationsList = document.getElementById('recommendations-list');
    const modelNameBadge = document.getElementById('model-name-badge');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get values
        const movie = document.getElementById('movie-input').value.trim();
        const model = document.getElementById('model-select').value;
        
        if (!movie) return;
        
        // Reset UI
        resultsContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        
        // Setup loading state
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        
        try {
            // Encode parameters
            const params = new URLSearchParams({ movie, model });
            
            // Make request
            const response = await fetch(`/recommend?${params.toString()}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch recommendations');
            }
            
            // Display results
            displayRecommendations(data);
            
        } catch (error) {
            // Display error
            errorMessage.textContent = error.message;
            errorContainer.classList.remove('hidden');
        } finally {
            // Remove loading state
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
    
    function displayRecommendations(data) {
        // Update model badge
        modelNameBadge.textContent = data.model || 'Machine Learning Model';
        
        // Clear previous results
        recommendationsList.innerHTML = '';
        
        // Add new results with staggering animation
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach((movie, index) => {
                const li = document.createElement('li');
                li.textContent = movie;
                li.style.opacity = '0';
                li.style.transform = 'translateY(15px)';
                li.style.transition = 'opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1), transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
                
                recommendationsList.appendChild(li);
                
                // Staggered animation matching the modern aesthetic
                setTimeout(() => {
                    li.style.opacity = '1';
                    li.style.transform = 'translateY(0)';
                }, 50 + (index * 100)); // Slight initial delay for smooth sequence
            });
        } else {
            const li = document.createElement('li');
            li.textContent = "No recommendations found. Try another movie.";
            recommendationsList.appendChild(li);
        }
        
        // Show container
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results on mobile devices softly
        if (window.innerWidth <= 640) {
            setTimeout(() => {
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }
    }
});
