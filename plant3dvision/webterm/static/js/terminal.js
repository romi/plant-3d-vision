document.addEventListener('DOMContentLoaded', function() {
    // Terminal Setup
    const term = new Terminal({
        cursorBlink: true,
        theme: {
            background: '#000000',
            foreground: '#ffffff'
        }
    });

    const fitAddon = new FitAddon.FitAddon();
    term.loadAddon(fitAddon);

    // Initialize terminal
    term.open(document.getElementById('terminal'));
    fitAddon.fit();

    // Socket.io connection
    const socket = io();
    let commandBuffer = '';

    socket.on('connect', () => {
        term.write('\r\n$ ');
    });

    socket.on('terminal_output', (data) => {
        if (data.error) {
            term.write('\r\nError: ' + data.error + '\r\n$ ');
        } else {
            term.write(data.output);
            if (!data.output.endsWith('$ ')) {
                term.write('\r\n$ ');
            }
        }
    });

    term.onKey(({ key, domEvent }) => {
        // Handle Enter
        if (domEvent.keyCode === 13) {
            if (commandBuffer.trim()) {
                socket.emit('terminal_input', { command: commandBuffer });
                commandBuffer = '';
            }
            term.write('\r\n');
        }
        // Handle Backspace
        else if (domEvent.keyCode === 8) {
            if (commandBuffer.length > 0) {
                commandBuffer = commandBuffer.slice(0, -1);
                term.write('\b \b');
            }
        }
        // Handle regular input
        else {
            commandBuffer += key;
            term.write(key);
        }
    });

    // Login/Logout Handling
    const loginForm = document.getElementById('login-form');
    const loginContainer = document.getElementById('login-container');
    const mainContainer = document.getElementById('main-container');
    const logoutBtn = document.getElementById('logout-btn');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();

            if (data.success) {
                loginContainer.style.display = 'none';
                mainContainer.style.display = 'block';
                socket.connect();
            } else {
                alert('Login failed');
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed');
        }
    });

    logoutBtn.addEventListener('click', async () => {
        try {
            await fetch('/logout', { method: 'POST' });
            loginContainer.style.display = 'block';
            mainContainer.style.display = 'none';
            socket.disconnect();
        } catch (error) {
            console.error('Logout error:', error);
        }
    });

    // Handle window resize
    window.addEventListener('resize', () => {
        fitAddon.fit();
    });
});