chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "predict") {
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text: request.text })
    })
    .then(response => {
      if (!response.ok) {
        return response.text().then(text => { throw new Error(text) });
      }
      return response.json();
    })
    .then(data => {
      // Send all relevant fields from the Flask response back
      sendResponse({
        prediction: {
          label: data.prediction_label,
          confidenceScore: data.confidence_score,
          confidenceLabel: data.confidence_label,
          persuasionScore: data.persuasion_score,
          persuasionLabel: data.persuasion_label,
          combinedLabel: data.combined_label
        }
      });
    })
    .catch(error => {
      sendResponse({ error: error.toString() });
    });

    return true;  // Keep message channel open for async response
  }
});
