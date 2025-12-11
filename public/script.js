document.addEventListener('DOMContentLoaded', () => {

    // 1. Dataset Gallery Rendering
    const galleryGrid = document.getElementById('gallery-grid');
    if (galleryGrid) {
        fetch('predictions.json')
            .then(response => response.json())
            .then(data => {
                // Only take first 8 items for the showcase
                const showcaseData = data.slice(0, 8);

                showcaseData.forEach(item => {
                    const figure = document.createElement('div');
                    figure.className = 'figure-card';
                    figure.innerHTML = `
                        <div class="figure-img-container">
                            <img src="${item.image}" alt="${item.label}" loading="lazy">
                        </div>
                        <div class="figure-caption">Figure: ${item.label}</div>
                    `;
                    galleryGrid.appendChild(figure);
                });
            })
            .catch(err => console.error('Error loading galaxy data:', err));
    }

    // 2. Interactive Demo Logic (Mock)
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultArea = document.getElementById('demo-result');
    const resultImg = document.getElementById('result-img');
    const resultBars = document.getElementById('result-bars');

    if (dropZone && fileInput) {
        // Handle click
        dropZone.addEventListener('click', () => fileInput.click());

        // Handle file select
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Handle Drag & Drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--accent-color)';
            dropZone.style.backgroundColor = '#f0f4f8';
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#eeeeee';
            dropZone.style.backgroundColor = '#fafafa';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#eeeeee';
            dropZone.style.backgroundColor = '#fafafa';

            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    function handleFile(file) {
        // 1. Show Image
        const reader = new FileReader();
        reader.onload = (e) => {
            resultImg.src = e.target.result;
            resultArea.classList.add('active');

            // 2. Mock Prediction (Simulate Network Delay)
            resultBars.innerHTML = '<p style="color:#666">Analyzing image...</p>';

            setTimeout(() => {
                const mockPredictions = [
                    { label: 'ALB (Albacore Tuna)', conf: 92 },
                    { label: 'YFT (Yellowfin Tuna)', conf: 5 },
                    { label: 'BET (Bigeye Tuna)', conf: 3 }
                ];

                renderBars(mockPredictions);

                // 3. Draw Simulated Bounding Box (SimpleFishNet Demo)
                drawBoundingBox(resultImg, 'ALB', 0.92);
            }, 800);
        };
        reader.readAsDataURL(file);
    }

    function drawBoundingBox(imgElement, label, conf) {
        // Ensure parent is relative
        const parent = imgElement.parentElement;
        parent.style.position = 'relative';
        parent.style.display = 'inline-block'; // Shrink to fit image

        // Clear existing boxes
        const existing = parent.querySelectorAll('.bounding-box');
        existing.forEach(el => el.remove());

        // Create Box
        const box = document.createElement('div');
        box.className = 'bounding-box';

        // Mock geometry (center-ish)
        // In a real app, these would come from the ONNX model output
        // x, y, w, h in %
        const top = 20 + Math.random() * 10;
        const left = 20 + Math.random() * 10;
        const width = 40 + Math.random() * 20;
        const height = 30 + Math.random() * 20;

        box.style.top = `${top}%`;
        box.style.left = `${left}%`;
        box.style.width = `${width}%`;
        box.style.height = `${height}%`;

        // Label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'box-label';
        labelDiv.innerText = `${label} ${(conf * 100).toFixed(0)}%`;
        box.appendChild(labelDiv);

        parent.appendChild(box);
    }

    function renderBars(predictions) {
        resultBars.innerHTML = '';
        predictions.forEach(p => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label">
                    <span>${p.label}</span>
                    <span>${p.conf}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: 0%"></div>
                </div>
            `;
            resultBars.appendChild(row);

            // Animate
            setTimeout(() => {
                row.querySelector('.bar-fill').style.width = `${p.conf}%`;
            }, 50);
        });
    }

    // 3. Smooth Scrolling for Nav
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

});
