document.querySelector("#predict-btn").addEventListener("click", async () => {
  const input = document.querySelector("#text-input").value;

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: input })
    });

    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      // If response is JSON, parse it as JSON
      const data = await response.json();
      console.log("Parsed JSON:", data);
      document.querySelector("#output").textContent = "Prediction: " + data.prediction;
    } else {
      // If not JSON, get raw text (likely an error page)
      const text = await response.text();
      console.error("Server returned non-JSON response:", text);
      document.querySelector("#output").textContent = "Server error occurred — check console for details.";
    }
  } catch (err) {
    document.querySelector("#output").textContent = "Something went wrong: " + err;
  }
});
