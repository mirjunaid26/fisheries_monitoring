document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.card[data-modal]');
    const closeButtons = document.querySelectorAll('.modal-close');
    const body = document.body;

    cards.forEach(card => {
        card.addEventListener('click', () => {
            const modalId = card.getAttribute('data-modal');
            const modal = document.getElementById(modalId);
            if (modal) {
                modal.classList.add('active');
                body.style.overflow = 'hidden'; // Prevent scrolling
            }
        });
    });

    closeButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const modal = btn.closest('.modal-overlay');
            closeModal(modal);
        });
    });

    // Close on click outside
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeModal(overlay);
            }
        });
    });

    function closeModal(modal) {
        modal.classList.remove('active');
        body.style.overflow = '';
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const activeModal = document.querySelector('.modal-overlay.active');
            if (activeModal) closeModal(activeModal);
        }
    });

    // Gallery Render Logic
    const galleryGrid = document.getElementById('gallery');
    if (galleryGrid) {
        fetch('predictions.json')
            .then(response => response.json())
            .then(data => {
                data.forEach((item, index) => {
                    const card = document.createElement('div');
                    card.className = 'gallery-item';
                    
                    const isCorrect = item.prediction === item.label;
                    const statusClass = isCorrect ? 'label-correct' : 'label-wrong';
                    const percent = Math.round(item.confidence * 100);

                    // Artificially delay animation for effect
                    const delay = index * 100; 

                    card.innerHTML = `
                        <div class="gallery-img-container">
                            <img src="${item.image}" alt="${item.label}" loading="lazy">
                        </div>
                        <div class="gallery-info">
                            <div class="prediction-label ${statusClass}">
                                <span>${item.prediction}</span>
                                <span>${percent}%</span>
                            </div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width: 0%"></div>
                            </div>
                        </div>
                    `;

                    galleryGrid.appendChild(card);

                    // Animate bar after render
                    setTimeout(() => {
                        const bar = card.querySelector('.confidence-bar-fill');
                        if(bar) bar.style.width = `${percent}%`;
                    }, 100 + delay);
                });
            })
            .catch(err => console.error('Error loading gallery:', err));
    }
});
