// Main landing page interactivity

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // FAQ accordion animation
    document.querySelectorAll('.faq-item summary').forEach(summary => {
        summary.addEventListener('click', (e) => {
            const details = summary.parentElement;
            const isOpen = details.hasAttribute('open');
            // Close others
            document.querySelectorAll('.faq-item[open]').forEach(item => {
                if (item !== details) item.removeAttribute('open');
            });
        });
    });

    // Navbar background on scroll
    const nav = document.querySelector('nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.classList.add('bg-dark/95');
        } else {
            nav.classList.remove('bg-dark/95');
        }
    });
});
