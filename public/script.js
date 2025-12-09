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
});
