document.addEventListener('DOMContentLoaded', () => {
    const pills = document.querySelectorAll('.demo-pill-row span');
    pills.forEach((pill, index) => {
        pill.style.transform = 'translateY(10px)';
        pill.style.opacity = '0';
        pill.animate([
            { transform: 'translateY(10px)', opacity: 0 },
            { transform: 'translateY(0)', opacity: 1 }
        ], {
            duration: 550,
            delay: index * 120,
            fill: 'forwards',
            easing: 'ease-out'
        });
    });

    const bars = document.querySelectorAll('.bar');
    bars.forEach((bar, index) => {
        bar.animate([
            { transform: 'scaleY(0.15)', opacity: 0.2 },
            { transform: 'scaleY(1)', opacity: 1 }
        ], {
            duration: 800,
            delay: 120 * index,
            fill: 'forwards',
            easing: 'cubic-bezier(0.22, 1, 0.36, 1)'
        });
    });
});