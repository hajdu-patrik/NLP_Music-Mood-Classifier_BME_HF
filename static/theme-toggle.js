// This is an IIFE (Immediately Invoked Function Expression)
(function() {
    // Find the toggle button in the DOM
    const toggleButton = document.getElementById('theme-toggle');
    
    // Check if a theme is already saved in the browser's local storage
    const currentTheme = localStorage.getItem('theme');

    // Apply the saved theme on load
    // By default, the site is dark. We ONLY add the class if the theme is 'light'.
    if (currentTheme === 'light') {
        document.body.classList.add('light-mode');
    }

    // Button click listener
    toggleButton.addEventListener('click', function() {
        // Toggle the .light-mode class on the <body> element
        document.body.classList.toggle('light-mode');
        
        let theme = 'dark'; // Default to dark
        // Check if the body now has the .light-mode class
        if (document.body.classList.contains('light-mode')) {
            theme = 'light';
        }
        
        // Save the user's preference to local storage
        localStorage.setItem('theme', theme);
    });
})();