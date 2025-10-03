document.getElementById('addUserForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData();
    formData.append('full_name', document.getElementById('full_name').value);
    formData.append('username', document.getElementById('username').value);
    formData.append('password', document.getElementById('password').value);

    fetch('/admin/add_user', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (data.success) {
                resultDiv.className = 'result success';
                resultDiv.textContent = 'User added successfully!';
                document.getElementById('addUserForm').reset();
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
