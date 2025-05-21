
chrome.runtime.onInstalled.addListener(() => {
    console.log('Extension installed');
  });
  
  // Example: listen for messages from popup.js and fetch data from API
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "predict") {
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: request.text})
      })
      .then(response => response.json())
      .then(data => sendResponse({prediction: data.prediction}))
      .catch(error => sendResponse({error: error.toString()}));
  
      // Return true to indicate async response
      return true;
    }
  });
  