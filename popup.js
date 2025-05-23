document.addEventListener("DOMContentLoaded", function () {
  const analyzeBtn = document.querySelector("#analyzeBtn");
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", () => {
      const input = document.querySelector("#inputText").value;

      fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: input })
      })
        .then(response => response.json())
        .then(data => {
          console.log("Prediction response:", data);
          document.querySelector("#confidenceLevel").textContent = data.confidence_label || "N/A";
          document.querySelector("#accuracyLevel").textContent = data.prediction_label || "N/A";
          document.querySelector("#finalLabel").textContent = data.combined_label || "N/A";
          document.querySelector("#results").classList.remove("d-none");
        })
        .catch(error => {
          alert("Error connecting to backend: " + error.message);
          console.error("Fetch error:", error);
        });
    });
  } else {
    console.error("Button #analyzeBtn not found in the DOM.");
  }
});
