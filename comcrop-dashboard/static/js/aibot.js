document.addEventListener('DOMContentLoaded', function() {
    const aiContent = document.getElementById('ai-bot-content');
    if (!aiContent) return;

    const queryInput = aiContent.querySelector('.query-input');
    const submitBtn = aiContent.querySelector('.submit-button');
    const responseDiv = aiContent.querySelector('.ai-response');
    const exampleBtns = aiContent.querySelectorAll('.example-btn');

    // Handle submissions
    async function handleSubmit() {
        const question = queryInput.value.trim();
        if (!question) return;
    
        // Clear previous response but keep model info
        const modelInfo = document.querySelector('.model-info').textContent;
        responseDiv.innerHTML = '<div class="loading">Processing your question...</div>';
        submitBtn.disabled = true;
    
        try {
            const startTime = Date.now();
            const response = await fetch('/api/ai-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: question })
            });
    
            const data = await response.json();
            const endTime = Date.now();
    
            // Update all fields
            document.querySelector('.model-info').textContent = data.model || modelInfo;
            document.querySelector('.response-time').textContent = 
                data.response_time || `${endTime - startTime}ms`;
            
            if (data.success) {
                responseDiv.innerHTML = `
                    <div class="response-text">${data.response.replace(/\n/g, '<br>')}</div>
                    <div class="response-meta">
                        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                    </div>
                `;
            } else {
                responseDiv.innerHTML = `<div class="error">${data.error || 'Error processing request'}</div>`;
            }
        } catch (error) {
            console.error('Error:', error);
            responseDiv.innerHTML = `<div class="error">Network error. Please try again.</div>`;
        } finally {
            submitBtn.disabled = false;
        }
    }

    // Event listeners
    submitBtn.addEventListener('click', handleSubmit);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSubmit();
    });

    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            queryInput.value = btn.textContent;
            handleSubmit();
        });
    });

    // Initial message
    responseDiv.innerHTML = `
        <div class="welcome-message">
            <p>Ask me about:</p>
            <ul>
                <li>Current water needs</li>
                <li>Demand forecasts</li>
                <li>Farm operation advice</li>
            </ul>
        </div>
    `;
});