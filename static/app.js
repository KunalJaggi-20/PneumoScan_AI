document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const btnDemoNormal = document.getElementById("btn-demo-normal");
    const btnDemoPneumonia = document.getElementById("btn-demo-pneumonia");
    const fileUpload = document.getElementById("file-upload");
    const dropzone = document.getElementById("dropzone");
    const uploadStatusText = document.getElementById("upload-status-text");
    const btnAnalyze = document.getElementById("btn-analyze");
    
    const emptyState = document.getElementById("empty-state");
    const loaderContainer = document.getElementById("loader-container");
    const resultsView = document.getElementById("results-view");
    
    const resultClass = document.getElementById("result-class");
    const resultConfidence = document.getElementById("result-confidence");
    const imgOriginal = document.getElementById("img-original");
    const imgHeatmap = document.getElementById("img-heatmap");
    const explanationText = document.getElementById("explanation-text");

    let selectedFile = null;
    let selectedDemo = null;

    // --- State Handlers ---
    
    function resetStates() {
        selectedFile = null;
        selectedDemo = null;
        
        btnDemoNormal.classList.remove("active");
        btnDemoPneumonia.classList.remove("active");
        
        uploadStatusText.textContent = "Choose file or drag here";
        fileUpload.value = "";
        
        btnAnalyze.disabled = true;
    }

    function selectDemo(demoName, buttonElement) {
        resetStates();
        selectedDemo = demoName;
        buttonElement.classList.add("active");
        btnAnalyze.disabled = false;
        
        // Show demo image preview in empty state
        const iconSvg = emptyState.querySelector(".empty-icon");
        if (iconSvg) iconSvg.style.display = "none";
        
        const emptyTitle = emptyState.querySelector("h3");
        if (emptyTitle) emptyTitle.textContent = `Demo: ${demoName === "demo_normal" ? "Normal" : "Pneumonia"} Radiograph Loaded`;
        
        const emptyDesc = emptyState.querySelector("p");
        if (emptyDesc) emptyDesc.textContent = "Click 'Run AI Analysis' to process this image and compute diagnostic attention maps.";
    }

    function selectFile(file) {
        resetStates();
        selectedFile = file;
        uploadStatusText.textContent = file.name;
        btnAnalyze.disabled = false;

        const iconSvg = emptyState.querySelector(".empty-icon");
        if (iconSvg) iconSvg.style.display = "none";
        
        const emptyTitle = emptyState.querySelector("h3");
        if (emptyTitle) emptyTitle.textContent = "Custom Image Loaded";
        
        const emptyDesc = emptyState.querySelector("p");
        if (emptyDesc) emptyDesc.textContent = `Selected: "${file.name}". Click 'Run AI Analysis' to submit this file to the neural network.`;
    }

    // --- Event Listeners ---

    // 1. Demo Buttons Click
    btnDemoNormal.addEventListener("click", () => {
        selectDemo("demo_normal", btnDemoNormal);
    });

    btnDemoPneumonia.addEventListener("click", () => {
        selectDemo("demo_pneumonia", btnDemoPneumonia);
    });

    // 2. File Upload Input
    fileUpload.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    });

    // 3. Drag and Drop Zone
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });

    // 4. Submit & Analyze Request
    btnAnalyze.addEventListener("click", async () => {
        if (!selectedFile && !selectedDemo) return;

        // Reset UI views
        emptyState.style.display = "none";
        resultsView.style.display = "none";
        loaderContainer.style.display = "block";
        btnAnalyze.disabled = true;

        // Build Payload
        const formData = new FormData();
        if (selectedFile) {
            formData.append("image", selectedFile);
        } else if (selectedDemo) {
            formData.append("demo_name", selectedDemo);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // Update prediction badge classes
                resultsView.classList.remove("normal-predicted", "pneumonia-predicted");
                if (result.prediction === "Normal") {
                    resultsView.classList.add("normal-predicted");
                } else {
                    resultsView.classList.add("pneumonia-predicted");
                }

                // Render Results
                resultClass.textContent = result.prediction.toUpperCase();
                resultConfidence.textContent = `${(result.confidence * 100).toFixed(1)}% Confidence`;
                imgOriginal.src = result.original_image;
                imgOriginal.style.display = "block";
                imgOriginal.alt = `Original chest X-ray image: ${result.prediction}`;
                imgHeatmap.src = result.heatmap_image;
                imgHeatmap.style.display = "block";
                imgOriginal.alt = `Model CAM heatmap: ${result.prediction}`;

                explanationText.innerHTML = result.explanation;

                // Toggle visibility
                loaderContainer.style.display = "none";
                resultsView.style.display = "block";
            } else {
                alert(`Error during analysis: ${result.error}`);
                loaderContainer.style.display = "none";
                emptyState.style.display = "block";
            }
        } catch (error) {
            console.error("Analysis Request Failed:", error);
            alert("Unable to communicate with the AI Server. Please ensure the Python server is running.");
            loaderContainer.style.display = "none";
            emptyState.style.display = "block";
        } finally {
            btnAnalyze.disabled = false;
        }
    });
});
