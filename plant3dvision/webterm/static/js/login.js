// Simple form validation
document.addEventListener('DOMContentLoaded', function () {
    const loginForm = document.querySelector('form');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');

    loginForm.addEventListener('submit', function (e) {
        // Reset previous error styling
        usernameInput.style.borderColor = '';
        passwordInput.style.borderColor = '';

        let isValid = true;

        // Simple validation
        if (!usernameInput.value.trim()) {
            usernameInput.style.borderColor = '#ff5757';
            isValid = false;
        }

        if (!passwordInput.value.trim()) {
            passwordInput.style.borderColor = '#ff5757';
            isValid = false;
        }

        if (!isValid) {
            e.preventDefault();
        }
    });
});