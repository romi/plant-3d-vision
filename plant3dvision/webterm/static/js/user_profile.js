document.getElementById('changePasswordForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const newPassword = document.getElementById('new_password').value;
    const confirmNewPassword = document.getElementById('confirm_new_password').value;

    // Check if new passwords match
    if (newPassword !== confirmNewPassword) {
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.className = 'result error';
        resultDiv.textContent = 'Error: New passwords do not match';
        return;
    }

    const formData = new FormData();
    formData.append('current_password', document.getElementById('current_password').value);
    formData.append('new_password', newPassword);

    fetch('/user/change_password', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (data.success) {
                resultDiv.className = 'result success';
                resultDiv.textContent = 'Password changed successfully!';
                document.getElementById('changePasswordForm').reset();
            } else {
                resultDiv.className = 'result error';
                resultDiv.textContent = 'Error: ' + data.error;
            }
        })
        .catch(error => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result error';
            resultDiv.textContent = 'Error: ' + error.message;
        });
});

function togglePasswordVisibility(inputId) {
    const passwordInput = document.getElementById(inputId);
    const toggleIcon = passwordInput.nextElementSibling.querySelector('i');

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        toggleIcon.classList.remove('bi-eye');
        toggleIcon.classList.add('bi-eye-slash');
    } else {
        passwordInput.type = "password";
        toggleIcon.classList.remove('bi-eye-slash');
        toggleIcon.classList.add('bi-eye');
    }
}